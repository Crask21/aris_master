
from datetime import timedelta
import math
import shutil
from diffusers.utils import is_tensorboard_available, logging
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from huggingface_hub import upload_folder
from waste_diffuser.parse_args import parse_args
from waste_diffuser.dataloader_interface import dataloaderInterface

from accelerate.logging import get_logger
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, InitProcessGroupKwargs
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import diffusers
import datasets
import logging

from diffusers import UNet2DModel, UNet2DConditionModel, AutoencoderKL, DDIMPipeline, DDPMPipeline
from diffusers.schedulers import DDIMScheduler
import inspect
from tqdm.auto import tqdm
import torch.nn.functional as F
import sys
from pathlib import Path
import time
from utils.pushover import send_notification

from summarize_training.save_config import save_args_as_config
from summarize_training.generate_config_summary import generate_data_summary_from_config
from waste_diffuser.pipeline import Pipeline
from waste_diffuser.masked_pipeline import MaskedPipeline

class training:
    def __init__(self):
        self.args = parse_args()
        self.config = UNet2DModel.load_config(self.args.config_path)
        # --------------------------------- AP NOTES --------------------------------- #
        '''
        Wait till Andreas was changed it to work with the new dataloader interface.'''
        try:
            generate_data_summary_from_config(config_path=self.args.config_path) # notes.md
        except Exception as e:
            print(f"WARNING: Failed to generate data summary: {e}", file=sys.stderr)
        # ------------------------------- AP NOTES END ------------------------------- #
        self.remaining_time_hms = "Unknown"
        dl_interface = dataloaderInterface(self.args.config_path, preview=False)
        self.dataloader = dl_interface.dataloader
        
        if self.config["logging"]["debug"]:
            self.display_batch(next(iter(self.dataloader)))
        
        #Initialize accelerator and logger
        self.initialize()
        #Initialize model
        self.config["model"]["num_class_embeds"] = dl_interface.num_classes
        self.config["model"]["sample_size"] = self.config["data"]["resolution"] // 8 if self.config["vae"]["use_vae"] else self.config["data"]["resolution"] # because of 3 downsamplings by factor 2 in the UNet architecture
        print(f"Model will be trained with {self.config['model']['num_class_embeds']} class embeds and sample size {self.config['model']['sample_size']}.")
        model = UNet2DModel.from_config(self.config["model"])
        self.num_classes = self.config["model"]["num_class_embeds"]
        
        # Create EMA for the model.
        if self.config["diffusion_parameters"]["use_ema"]:
            self.ema_model = EMAModel(
                model.parameters(),
                decay=self.config["diffusion_parameters"]["ema_max_decay"],
                use_ema_warmup=True,
                inv_gamma=self.config["diffusion_parameters"]["ema_inv_gamma"],
                power=self.config["diffusion_parameters"]["ema_power"],
                model_cls=UNet2DModel,
                model_config=model.config,
            )
            
        #Mixed precision.
        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
            self.config["diffusion_parameters"]["mixed_precision"] = self.accelerator.mixed_precision
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16
            self.config["diffusion_parameters"]["mixed_precision"] = self.accelerator.mixed_precision
            
        # Initialize the scheduler
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=self.config["diffusion_parameters"]["ddpm_num_steps"],
            beta_schedule=self.config["diffusion_parameters"]["ddpm_beta_schedule"],
            prediction_type=self.config["diffusion_parameters"]["prediction_type"],
        )
        
        # Initialize the optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config["hyperparameters"]["learning_rate"],
            betas=(self.config["diffusion_parameters"]["adam_beta1"], self.config["diffusion_parameters"]["adam_beta2"]),
            weight_decay=self.config["diffusion_parameters"]["adam_weight_decay"],
            eps=self.config["diffusion_parameters"]["adam_epsilon"],
        )
        
        # Initialize the learning rate scheduler
        lr_scheduler = get_scheduler(
            self.config["diffusion_parameters"]["lr_scheduler"],
            optimizer=optimizer,
            num_warmup_steps=self.config["diffusion_parameters"]["lr_warmup_steps"] * self.config["diffusion_parameters"]["gradient_accumulation_steps"],
            num_training_steps=(len(self.dataloader) * self.config["hyperparameters"]["epochs"]),
        )
        
        
        # Prepare everything with `accelerator` and ema.
        self.model, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            model, optimizer, self.dataloader, lr_scheduler
        )
        if self.config["diffusion_parameters"]["use_ema"]:
            self.ema_model.to(self.accelerator.device)
        
        
        
        if self.accelerator.is_main_process:
            run = os.path.split(__file__)[-1].split(".")[0]
            self.accelerator.init_trackers(run)
        
        
        # Load checkpoint if specified
        self.load_checkpoint()
        # Load VAE if specified
        # if config["vae"]["use_vae"]:
        #     self.vae = AutoencoderKL.from_pretrained(config["vae"]["vae_model_path"]).to(self.accelerator.device)
        
        
    
    def initialize(self):
        '''
        Intialize the accelerator and logger
        '''
        self.logger = get_logger(__name__, log_level="INFO")
        
        logging_dir = os.path.join(self.config["logging"]["output_dir"], "logs")
        accelerator_project_config = ProjectConfiguration(project_dir=self.config["logging"]["output_dir"], logging_dir=logging_dir)

        kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))  # a big number for high resolution or big dataset
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.config["diffusion_parameters"]["gradient_accumulation_steps"],
            mixed_precision=self.config["diffusion_parameters"]["mixed_precision"],
            log_with=self.config["logging"]["logger"],
            project_config=accelerator_project_config,
            kwargs_handlers=[kwargs],
        )
        if self.config["logging"]["logger"] == "tensorboard":
            if not is_tensorboard_available():
                raise ImportError("Make sure to install tensorboard if you want to use it for logging during training.")

        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        self.logger.info(self.accelerator.state, main_process_only=False)
        
        # Set the verbosity level 
        datasets.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
        
        # Create output dir
        if self.config["logging"]["output_dir"] is not None:
            os.makedirs(self.config["logging"]["output_dir"], exist_ok=True)
    
    def load_checkpoint(self):
        '''
        Storage function for loading a checkpoint if specified in the arguments.
        '''
        # Set training state
        total_batch_size = self.config['hyperparameters']['batch_size'] * self.accelerator.num_processes * self.config["diffusion_parameters"]["gradient_accumulation_steps"]
        self.num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.config["diffusion_parameters"]["gradient_accumulation_steps"])
        max_train_steps = self.config["hyperparameters"]["epochs"] * self.num_update_steps_per_epoch

        self.logger.info("***** Running training *****")
        self.logger.info(f"  Num examples = {len(self.train_dataloader.dataset)}")
        self.logger.info(f"  Num Epochs = {self.config['hyperparameters']['epochs']}")
        self.logger.info(f"  Instantaneous batch size per device = {self.config['hyperparameters']['batch_size']}")
        self.logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        self.logger.info(f"  Gradient Accumulation steps = {self.config['diffusion_parameters']['gradient_accumulation_steps']}")
        self.logger.info(f"  Total optimization steps = {max_train_steps}")

        self.global_step = 0
        self.first_epoch = 0
        #Load checkpoint
        if self.config["logging"]["resume_from_checkpoint"]:
            if self.config["logging"]["resume_from_checkpoint"] != "latest":
                path = os.path.basename(self.config["logging"]["resume_from_checkpoint"])
            else:
                # Get the most recent checkpoint
                dirs = os.listdir(self.config["logging"]["output_dir"])
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None
            if path is None:
                self.accelerator.print(
                    f"Checkpoint '{self.config['logging']['resume_from_checkpoint']}' does not exist. Starting a new training run."
                )
                self.config["logging"]["resume_from_checkpoint"] = None
            else:
                self.accelerator.print(f"Resuming from checkpoint {path}")
                self.accelerator.load_state(os.path.join(self.config["logging"]["output_dir"], path))
                self.global_step = int(path.split("-")[1])

                self.resume_global_step = self.global_step * self.config["diffusion_parameters"]["gradient_accumulation_steps"]
                self.first_epoch = self.global_step // self.num_update_steps_per_epoch
                self.resume_step = self.resume_global_step % (self.num_update_steps_per_epoch * self.config["diffusion_parameters"]["gradient_accumulation_steps"])
                
                # # HOTFIX for when the num of epochs are adjusted. Then the scheduler needs to get a modified base_lr
                # total_steps = self.config["hyperparameters"]["epochs"] * self.num_update_steps_per_epoch
                # state_dict = self.lr_scheduler.state_dict()
                # state_dict["base_lrs"] = [self.lr_scheduler.get_last_lr()[0] * total_steps / (total_steps - self.global_step)]
                # self.lr_scheduler.load_state_dict(state_dict)
                    
    
    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        """
        Extract values from a 1-D numpy array for a batch of indices.

        :param arr: the 1-D numpy array.
        :param timesteps: a tensor of indices into the array to extract.
        :param broadcast_shape: a larger shape of K dimensions with the batch
                                dimension equal to the length of timesteps.
        :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
        """
        if not isinstance(arr, torch.Tensor):
            arr = torch.from_numpy(arr)
        res = arr[timesteps].float().to(timesteps.device)
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)
    
    def train(self):
        
        #print total parameters for unet
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total parameters in UNet: {total_params}")

        start_time = time.time()
        for epoch in range(self.first_epoch, self.config["hyperparameters"]["epochs"]):
            
            self.model.train()
            self.progress_bar = tqdm(total=self.num_update_steps_per_epoch)
            self.progress_bar.set_description(f"Epoch {epoch}")
            
            
            total_epoch_time = time.time()
            time_spent_loading = 0
            time_spent_in_model = 0
            time_spent_weight_update = 0
            time_other = 0
            time_save_checkpoint = 0
            
            tm = time.time()
            
            for step, batch in enumerate(self.train_dataloader):
                # Skip steps until we reach the resumed step
                if self.config["logging"]["resume_from_checkpoint"] and epoch == self.first_epoch and step < self.resume_step:
                    if step % self.config["diffusion_parameters"]["gradient_accumulation_steps"] == 0:
                        self.progress_bar.update(1)
                    continue
                
                class_labels = batch["class"]
                class_labels = class_labels.to(self.accelerator.device)
                clean_images = batch["image"].to(self.accelerator.device)
                masks = batch["masks"].to(self.accelerator.device) if "masks" in batch else None
                
                time_spent_loading += time.time() - tm
                tm = time.time()
                # Sample noise that we'll add to the images
                noise = torch.randn(clean_images.shape, dtype=self.weight_dtype, device=clean_images.device)
                bsz = clean_images.shape[0]
                
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=clean_images.device
                ).long()
                
                # Add noise to the clean images according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_images = self.noise_scheduler.add_noise(clean_images, noise, timesteps)

                # print(masks.shape if masks is not None else "No masks provided")
                #Concatenate masks if they exist
                if masks is not None:
                    #mask current shape is (B, H, W). We need to add a channel dimension and repeat it to match the number of channels in the image (3 for RGB)
                    masks = masks.unsqueeze(1) # (B, 1, H, W) 
                    noisy_images = torch.cat([noisy_images, masks], dim=1)
                
                
                # ----------------------------- model predictions ---------------------------- #
                with self.accelerator.accumulate(self.model):
                    
                    time_other += time.time() - tm
                    # Predict the noise residual
                    tm = time.time()
                    model_output = self.model(noisy_images, timesteps, class_labels=class_labels).sample
                    time_spent_in_model += time.time() - tm
                    

                    tm = time.time()
                    if self.config["diffusion_parameters"]["prediction_type"] == "epsilon":
                        loss = F.mse_loss(model_output.float(), noise.float())  # this could have different weights!
                    elif self.config["diffusion_parameters"]["prediction_type"] == "sample":
                        alpha_t = self._extract_into_tensor(
                            self.noise_scheduler.alphas_cumprod, timesteps, (clean_images.shape[0], 1, 1, 1)
                        )
                        snr_weights = alpha_t / (1 - alpha_t)
                        # use SNR weighting from distillation paper
                        loss = snr_weights * F.mse_loss(model_output.float(), clean_images.float(), reduction="none")
                        loss = loss.mean()
                    else:
                        raise ValueError(f"Unsupported prediction type: {self.config['diffusion_parameters']['prediction_type']}")
                    
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    time_spent_weight_update += time.time() - tm
                    
                # ------------------------------ SAVE CHECKPOINT ----------------------------- #
                tm = time.time()
                self.save_checkpoint(loss)
                time_save_checkpoint += time.time() - tm

                tm = time.time()
            print(f"Epoch {epoch} finished in {time.time() - total_epoch_time:.2f} seconds. Time spent loading data: {time_spent_loading:.2f} seconds. Time spent in model: {time_spent_in_model:.2f} seconds. Time spent updating weights: {time_spent_weight_update:.2f} seconds. Time spent saving checkpoint: {time_save_checkpoint:.2f} seconds. Time spent on other tasks: {time_other:.2f} seconds.")
            self.progress_bar.close()
            
            # --------------- Generate sample images for visual inspection --------------- #
            if epoch % self.config["logging"]["save_images_epochs"] == 0 or epoch == self.config["hyperparameters"]["epochs"] - 1:
                unet = self.accelerator.unwrap_model(self.model)
                images_processed = self.inference(unet,
                                                scheduler=self.noise_scheduler,
                                                )
                print(images_processed.shape)

                tracker = self.accelerator.get_tracker("tensorboard", unwrap=True)
                tracker.add_images("test_samples", images_processed, epoch)
            
            if epoch % self.config["logging"]["save_model_epochs"] == 0 or epoch == self.config["hyperparameters"]["epochs"] - 1:
                # save the model
                unet = self.accelerator.unwrap_model(self.model)

                if self.config["diffusion_parameters"]["use_ema"]:
                    self.ema_model.store(unet.parameters())
                    self.ema_model.copy_to(unet.parameters())

                pipeline = DDIMPipeline(
                    unet=unet,
                    scheduler=self.noise_scheduler,
                )

                pipeline.save_pretrained(self.config["logging"]["output_dir"])

                if self.config["diffusion_parameters"]["use_ema"]:
                    self.ema_model.restore(unet.parameters())
                        
            # -------------------------- Estimate remaining time ------------------------- #
            elapsed_time = time.time() - start_time
            avg_time_per_epoch = elapsed_time / (epoch + 1 - self.first_epoch)
            remaining_time = avg_time_per_epoch * (self.config["hyperparameters"]["epochs"] - epoch - 1)
            # remaining time in hh:mm:ss format
            self.remaining_time_hms = time.strftime("%d days %H:%M:%S", time.gmtime(remaining_time))
            # Estimated time that the model is expected to finish training
            estimated_finish_time = time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() + remaining_time))
            
        send_notification(
            title="Training complete",
            message=f"Training of {self.config['logging']['output_dir']} is complete! Total training time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}")
            
        
            
            
        self.accelerator.end_training()
    
    def train_vae(self):
        start_time = time.time()
        for epoch in range(self.first_epoch, self.config["hyperparameters"]["epochs"]):
            
            self.model.train()
            self.progress_bar = tqdm(total=self.num_update_steps_per_epoch)
            self.progress_bar.set_description(f"Epoch {epoch}")
            
            batch = next(iter(self.train_dataloader))
            images = batch["image"]
            
            
            for step, batch in enumerate(self.train_dataloader):
                # Skip steps until we reach the resumed step
                if self.config["logging"]["resume_from_checkpoint"] and epoch == self.first_epoch and step < self.resume_step:
                    if step % self.config["diffusion_parameters"]["gradient_accumulation_steps"] == 0:
                        self.progress_bar.update(1)
                    continue
                
                
                class_labels = batch["class"]
                class_labels = class_labels.to(self.accelerator.device)
                clean_images = batch["image"].to(self.accelerator.device)
                
                
                # # print(f"Clean images shape: {clean_images.shape}, dtype: {clean_images.dtype}")
                # # VAE ENCODING:
                # with torch.no_grad():
                #     clean_images = self.vae.encode(clean_images).latent_dist.sample()
                #     # Scale latents by VAE scaling factor (important!)
                #     clean_images = clean_images * self.vae.config.scaling_factor
                #     # print(f"Latent images shape: {clean_images.shape}, dtype: {clean_images.dtype}")
                    
                # Sample noise that we'll add to the images
                noise = torch.randn(clean_images.shape, dtype=self.weight_dtype, device=clean_images.device)
                bsz = clean_images.shape[0]
                
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=clean_images.device
                ).long()
                
                # Add noise to the clean images according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_images = self.noise_scheduler.add_noise(clean_images, noise, timesteps)
            
                
                
                # ----------------------------- model predictions ---------------------------- #
                with self.accelerator.accumulate(self.model):
                    # Predict the noise residual
                    model_output = self.model(noisy_images, timesteps, class_labels=class_labels).sample

                    if self.config["diffusion_parameters"]["prediction_type"] == "epsilon":
                        loss = F.mse_loss(model_output.float(), noise.float())  # this could have different weights!
                    elif self.config["diffusion_parameters"]["prediction_type"] == "sample":
                        alpha_t = self._extract_into_tensor(
                            self.noise_scheduler.alphas_cumprod, timesteps, (clean_images.shape[0], 1, 1, 1)
                        )
                        snr_weights = alpha_t / (1 - alpha_t)
                        # use SNR weighting from distillation paper
                        loss = snr_weights * F.mse_loss(model_output.float(), clean_images.float(), reduction="none")
                        loss = loss.mean()
                    else:
                        raise ValueError(f"Unsupported prediction type: {self.config['diffusion_parameters']['prediction_type']}")
                    
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    
                # ------------------------------ SAVE CHECKPOINT ----------------------------- #
                self.save_checkpoint(loss)
            self.progress_bar.close()
            
            # --------------- Generate sample images for visual inspection --------------- #
            if epoch % self.config["logging"]["save_images_epochs"] == 0 or epoch == self.config["hyperparameters"]["epochs"] - 1:
                unet = self.accelerator.unwrap_model(self.model)
                vae = AutoencoderKL.from_pretrained(self.config["vae"]["vae_model_path"]).to(self.accelerator.device)
                images_processed = self.inference(unet,
                                                scheduler=self.noise_scheduler,
                                                vae=vae
                                                )
                vae.to("cpu")

                tracker = self.accelerator.get_tracker("tensorboard", unwrap=True)
                tracker.add_images("test_samples", images_processed.transpose(0, 3, 1, 2), epoch)
            
            if epoch % self.config["logging"]["save_model_epochs"] == 0 or epoch == self.config["hyperparameters"]["epochs"] - 1:
                # save the model
                unet = self.accelerator.unwrap_model(self.model)

                if self.config["diffusion_parameters"]["use_ema"]:
                    self.ema_model.store(unet.parameters())
                    self.ema_model.copy_to(unet.parameters())

                pipeline = DDIMPipeline(
                    unet=unet,
                    scheduler=self.noise_scheduler,
                )

                pipeline.save_pretrained(self.config["logging"]["output_dir"])

                if self.config["diffusion_parameters"]["use_ema"]:
                    self.ema_model.restore(unet.parameters())
                    
            # -------------------------- Estimate remaining time ------------------------- #
            elapsed_time = time.time() - start_time
            avg_time_per_epoch = elapsed_time / (epoch + 1 - self.first_epoch)
            remaining_time = avg_time_per_epoch * (self.config["hyperparameters"]["epochs"] - epoch - 1)
            # remaining time in hh:mm:ss format
            self.remaining_time_hms = time.strftime("%H:%M:%S", time.gmtime(remaining_time))
            # Estimated time that the model is expected to finish training
            estimated_finish_time = time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() + remaining_time))
            # print(f"Remaining time: {remaining_time_hms}, Estimated finish time: {estimated_finish_time}")
            
            self.accelerator.end_training()    
        send_notification(
            title="Training complete",
            message=f"Training of {self.config['logging']['output_dir']} is complete! Total training time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}")
                
    def save_checkpoint(self, loss):
        # Checks if the accelerator has performed an optimization step behind the scenes
        if self.accelerator.sync_gradients:
            if self.config["diffusion_parameters"]["use_ema"]:
                self.ema_model.step(self.model.parameters())
            self.progress_bar.update(1)
            self.global_step += 1
            if self.global_step % self.config["logging"]["checkpointing_steps"] == 0:
                # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                if self.config["logging"]["checkpoints_total_limit"] is not None:
                    checkpoints = os.listdir(self.config["logging"]["output_dir"])
                    checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                    # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                    if len(checkpoints) >= self.config["logging"]["checkpoints_total_limit"]:
                        num_to_remove = len(checkpoints) - self.config["logging"]["checkpoints_total_limit"] + 1
                        removing_checkpoints = checkpoints[0:num_to_remove]

                        self.logger.info(
                            f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                        )
                        self.logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                        for removing_checkpoint in removing_checkpoints:
                            removing_checkpoint = os.path.join(self.config["logging"]["output_dir"], removing_checkpoint)
                            shutil.rmtree(removing_checkpoint)

                save_path = os.path.join(self.config["logging"]["output_dir"], f"checkpoint-{self.global_step}")
                self.accelerator.save_state(save_path)
                self.logger.info(f"Saved state to {save_path}")

        logs = {"loss": loss.detach().item(), "lr": self.lr_scheduler.get_last_lr()[0], "step": self.global_step, "time_remaining": self.remaining_time_hms}
        if self.config["diffusion_parameters"]["use_ema"]:
            logs["ema_decay"] = self.ema_model.cur_decay_value
        self.progress_bar.set_postfix(**logs)
        self.accelerator.log(logs, step=self.global_step)
        
    def inference(self, unet, scheduler=None, vae=None):
        
        if self.config["diffusion_parameters"]["use_ema"]:
            self.ema_model.store(unet.parameters())
            self.ema_model.copy_to(unet.parameters())

        pipeline = MaskedPipeline(
            unet=unet,
            scheduler=scheduler,
        )
        generator = torch.Generator(device=pipeline.device).manual_seed(0)
        # run pipeline in inference (sample random noise and denoise)
        # Get latents from pipeline

        print("Running inference with class conditioning.")
        # Create class labels for evaluation.
        class_labels = torch.from_numpy(np.linspace(0, self.num_classes - 0.01, self.config["hyperparameters"]["eval_batch_size"], dtype=int)).to(self.accelerator.device)
        
        #Make the mask a [B, 1, H, W] tensor with a square in the middle for now. The square will be 1 in the middle and 0 elsewhere, and the size of the square will be 1/3 of the image size. 
        masks = torch.zeros((self.config["hyperparameters"]["eval_batch_size"], 1, self.config["data"]["resolution"], self.config["data"]["resolution"]), device=self.accelerator.device)
        square_size = self.config["data"]["resolution"] // 3
        start = self.config["data"]["resolution"] // 2 - square_size // 2
        end = start + square_size
        masks[:, :, start:end, start:end] = 1.0
        #save mask as image for debugging
        from PIL import Image
        mask_image = (masks[0, 0].cpu().numpy() * 255).astype("uint8")
        Image.fromarray(mask_image).save("mask.png")
        # print("class_labels.shape:", class_labels.shape)
        latents = pipeline(
            generator=generator,
            batch_size=self.config["hyperparameters"]["eval_batch_size"],
            num_inference_steps=self.config["diffusion_parameters"]["ddpm_num_inference_steps"],
            output_type="latent",
            class_labels=class_labels,
            masks=masks,
            return_dict=False
        )[0]
        
        if vae is not None:
            print(f"Generated latents shape: {latents.shape}, dtype: {latents.dtype}")
            print("Decoding latents with VAE.")
            with torch.no_grad():
                # Unscale latents before decoding
                latents = latents / vae.config.scaling_factor
                latents = vae.decode(latents).sample.float().cpu().numpy()
            latents = latents.transpose(0, 2, 3, 1)  # Convert back to (B, H, W, C)
            print("Decoded images shape:", latents.shape, "Decoded images dtype:", latents.dtype, "Decoded images min:", latents.min(), "Decoded images max:", latents.max())
            

        if self.config["diffusion_parameters"]["use_ema"]:
            self.ema_model.restore(unet.parameters())
            
        images = latents
        self.logger.info(f"Generated latents shape: {images.shape}")
        
        # if type(images) == torch.Tensor:
        if type(images) == torch.Tensor:
            images = images.cpu().numpy()
        
        # denormalize the images (VAE outputs are in [-1, 1] range)
        images_processed = np.array(((images / 2 + 0.5).clip(0, 1) * 255).round()).astype("uint8")
        return images_processed
    
    def display_batch(self, batch, num_images=9):
        '''
        Function for displaying a batch of images from the dataloader.
        --batch: A batch of images from the dataloader.
        --num_images: The number of images to display from the batch.
        '''
        #DONE
        print(batch.keys())
        images = batch["image"]
        
        #if batch contains ["masks"], then overlay the masks on the images for visualization
        if "masks" in batch:
            masks = batch["masks"]
            print(masks.shape, images.shape) # for debugging, they are (B, H, W) and (B, C, H, W)
            # create a red mask for visualization
            red_masks = torch.zeros_like(images)
            red_masks[:, 0, :, :] = masks  # set the red channel to the masks
            # overlay the red masks on the images
            images = torch.clamp(images + red_masks, -1, 1)
            
        
        #limit to 3 channels if more (e.g., for latents)
        if images.shape[1] > 3:
            images = images[:, :3, :, :]
        
        # Denormalize images from [-1, 1] to [0, 1]
        images = (images + 1) / 2
        images = torch.clamp(images, 0, 1)
        # Create grid
        grid_size = int(np.ceil(np.sqrt(num_images)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
        axes = axes.flatten()
        
        for idx, ax in enumerate(axes):
            if idx < len(images):
                # Convert to numpy and transpose from CxHxW to HxWxC
                img = images[idx].permute(1, 2, 0).cpu().numpy()
                ax.imshow(img)
                ax.axis('off')
            else:
                ax.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        
        
    


if __name__ == "__main__":
    trainer = training()
    
    # ----------------------------------- TRAIN ---------------------------------- #
    send_notification(title="Training started", message=f"Chugga chugga!")
    
    import traceback
    try:
        if trainer.config["vae"]["use_vae"]:
            trainer.train_vae()
        else:
            trainer.train()
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        traceback.print_exc()
        send_notification(
            title="Training error",
            message=f"An error occurred during training of {trainer.output_dir}: {e}")
    
    # batch = next(iter(trainer.train_dataloader))
    # batch = trainer.inference(unet=trainer.accelerator.unwrap_model(trainer.model), scheduler=trainer.noise_scheduler)
    # trainer.display_batch(batch, num_images=16)
    
    
        