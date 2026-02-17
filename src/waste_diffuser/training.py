
from datetime import timedelta
import math
import shutil
from diffusers.utils import is_tensorboard_available, logging
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from huggingface_hub import upload_folder
import wandb
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

from diffusers import UNet2DModel, AutoencoderKL, DDIMPipeline, DDPMPipeline
from diffusers.schedulers import DDIMScheduler
import inspect
from tqdm.auto import tqdm
import torch.nn.functional as F
import sys
from pathlib import Path
import time
from utils.pushover import send_notification

from summarize_training.save_config import save_args_as_config
from summarize_training.generate_config_summary import generate_data_summary
from waste_diffuser.pipeline import Pipeline

class training:
    def __init__(self):
        self.args = parse_args()
        
        # --------------------------------- AP NOTES --------------------------------- #
        try:
            save_args_as_config(args=self.args, parser=None) # config.json
        except Exception as e:
            print(f"WARNING: Failed to save config: {e}", file=sys.stderr)
        
        try:
            generate_data_summary(dataset_path=self.args.train_data_dir, output_path=self.args.output_dir, comment=self.args.comment) # notes.md
        except Exception as e:
            print(f"WARNING: Failed to generate data summary: {e}", file=sys.stderr)
        # ------------------------------- AP NOTES END ------------------------------- #
        
        dl_interface = dataloaderInterface(self.args.config_path)
        self.dataloader = dl_interface.dataloader
        print("Dataloader loaded successfully.")
        
        # self.display_batch(next(iter(self.dataloader)))
        
        #Initialize accelerator and logger
        self.initialize()
        #Initialize model
        config = UNet2DModel.load_config(self.args.config_path)
        config["model"]["num_class_embeds"] = dl_interface.num_classes
        model = UNet2DModel.from_config(config["model"])
        self.num_classes = config["model"]["num_class_embeds"]
        
        # Create EMA for the model.
        if self.args.use_ema:
            self.ema_model = EMAModel(
                model.parameters(),
                decay=self.args.ema_max_decay,
                use_ema_warmup=True,
                inv_gamma=self.args.ema_inv_gamma,
                power=self.args.ema_power,
                model_cls=UNet2DModel,
                model_config=model.config,
            )
            
        #Mixed precision.
        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
            self.args.mixed_precision = self.accelerator.mixed_precision
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16
            self.args.mixed_precision = self.accelerator.mixed_precision
            
        # Initialize the scheduler
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=self.args.ddpm_num_steps,
            beta_schedule=self.args.ddpm_beta_schedule,
            prediction_type=self.args.prediction_type,
        )
        
        # Initialize the optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            weight_decay=self.args.adam_weight_decay,
            eps=self.args.adam_epsilon,
        )
        
        # Initialize the learning rate scheduler
        lr_scheduler = get_scheduler(
            self.args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.args.lr_warmup_steps * self.args.gradient_accumulation_steps,
            num_training_steps=(len(self.dataloader) * self.args.num_epochs),
        )
        # Prepare everything with `accelerator` and ema.
        self.model, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            model, optimizer, self.dataloader, lr_scheduler
        )
        if self.args.use_ema:
            self.ema_model.to(self.accelerator.device)
            
        if self.accelerator.is_main_process:
            run = os.path.split(__file__)[-1].split(".")[0]
            self.accelerator.init_trackers(run)
        
        # Load checkpoint if specified
        self.load_checkpoint()
        
        # Load VAE if specified
        if config["vae"]["use_vae"]:
            self.vae = AutoencoderKL.from_pretrained(config["vae"]["vae_model_path"]).to(self.accelerator.device)
        
        # ----------------------------------- TRAIN ---------------------------------- #
        try:
            if config["vae"]["use_vae"]:
                self.train_vae()
            else:
                self.train()
        except Exception as e:
            print(f"\n✗ Error during training: {e}")
            send_notification(
                title="Training error",
                message=f"An error occurred during training of {self.output_dir}: {e}")
    
    def initialize(self):
        '''
        Intialize the accelerator and logger
        '''
        self.logger = get_logger(__name__, log_level="INFO")
        
        logging_dir = os.path.join(self.args.output_dir, self.args.logging_dir)
        accelerator_project_config = ProjectConfiguration(project_dir=self.args.output_dir, logging_dir=logging_dir)

        kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))  # a big number for high resolution or big dataset
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            mixed_precision=self.args.mixed_precision,
            log_with=self.args.logger,
            project_config=accelerator_project_config,
            kwargs_handlers=[kwargs],
        )
        if self.args.logger == "tensorboard":
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
        if self.args.output_dir is not None:
            os.makedirs(self.args.output_dir, exist_ok=True)
    
    def load_checkpoint(self):
        '''
        Storage function for loading a checkpoint if specified in the arguments.
        '''
        # Set training state
        total_batch_size = self.args.train_batch_size * self.accelerator.num_processes * self.args.gradient_accumulation_steps
        self.num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.args.gradient_accumulation_steps)
        max_train_steps = self.args.num_epochs * self.num_update_steps_per_epoch

        self.logger.info("***** Running training *****")
        self.logger.info(f"  Num examples = {len(self.train_dataloader.dataset)}")
        self.logger.info(f"  Num Epochs = {self.args.num_epochs}")
        self.logger.info(f"  Instantaneous batch size per device = {self.args.train_batch_size}")
        self.logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        self.logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        self.logger.info(f"  Total optimization steps = {max_train_steps}")

        self.global_step = 0
        self.first_epoch = 0
        #Load checkpoint
        if self.args.resume_from_checkpoint:
            if self.args.resume_from_checkpoint != "latest":
                path = os.path.basename(self.args.resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = os.listdir(self.args.output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None
            if path is None:
                self.accelerator.print(
                    f"Checkpoint '{self.args.resume_from_checkpoint}' does not exist. Starting a new training run."
                )
                self.args.resume_from_checkpoint = None
            else:
                self.accelerator.print(f"Resuming from checkpoint {path}")
                self.accelerator.load_state(os.path.join(self.args.output_dir, path))
                self.global_step = int(path.split("-")[1])

                self.resume_global_step = self.global_step * self.args.gradient_accumulation_steps
                self.first_epoch = self.global_step // self.num_update_steps_per_epoch
                self.resume_step = self.resume_global_step % (self.num_update_steps_per_epoch * self.args.gradient_accumulation_steps)
                           
    def initialize_model(self, model_name_or_path, model_function):
        '''
        Initialize the model and move it to the accelerator device.
        --model_name_or_path: The name or path of the model to initialize.
        --model_function: The function to use for initializing the model (e.g., UNet2DModel.from_pretrained).
        '''
        config = UNet2DModel.load_config(self.args.model_config_name_or_path)
        model = UNet2DModel.from_config(config)
        
        model = model_function(model_name_or_path)
        return model.to(self.accelerator.device)       
    
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
        
        start_time = time.time()
        
        for epoch in range(self.first_epoch, self.args.num_epochs):
            
            self.model.train()
            self.progress_bar = tqdm(total=self.num_update_steps_per_epoch)
            self.progress_bar.set_description(f"Epoch {epoch}")
            for step, batch in enumerate(self.train_dataloader):
                # Skip steps until we reach the resumed step
                if self.args.resume_from_checkpoint and epoch == self.first_epoch and step < self.resume_step:
                    if step % self.args.gradient_accumulation_steps == 0:
                        self.progress_bar.update(1)
                    continue
                
                
                class_labels = batch["class"]
                class_labels = class_labels.to(self.accelerator.device)
                clean_images = batch["image"].to(self.accelerator.device)
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

                    if self.args.prediction_type == "epsilon":
                        loss = F.mse_loss(model_output.float(), noise.float())  # this could have different weights!
                    elif self.args.prediction_type == "sample":
                        alpha_t = self._extract_into_tensor(
                            self.noise_scheduler.alphas_cumprod, timesteps, (clean_images.shape[0], 1, 1, 1)
                        )
                        snr_weights = alpha_t / (1 - alpha_t)
                        # use SNR weighting from distillation paper
                        loss = snr_weights * F.mse_loss(model_output.float(), clean_images.float(), reduction="none")
                        loss = loss.mean()
                    else:
                        raise ValueError(f"Unsupported prediction type: {self.args.prediction_type}")
                    
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
            if epoch % self.args.save_images_epochs == 0 or epoch == self.args.num_epochs - 1:
                unet = self.accelerator.unwrap_model(self.model)
                images_processed = self.inference(unet,
                                                scheduler=self.noise_scheduler,
                                                )

                tracker = self.accelerator.get_tracker("tensorboard", unwrap=True)
                tracker.add_images("test_samples", images_processed.transpose(0, 3, 1, 2), epoch)
            
            if epoch % self.args.save_model_epochs == 0 or epoch == self.args.num_epochs - 1:
                # save the model
                unet = self.accelerator.unwrap_model(self.model)

                if self.args.use_ema:
                    self.ema_model.store(unet.parameters())
                    self.ema_model.copy_to(unet.parameters())

                pipeline = DDIMPipeline(
                    unet=unet,
                    scheduler=self.noise_scheduler,
                )

                pipeline.save_pretrained(self.args.output_dir)

                if self.args.use_ema:
                    self.ema_model.restore(unet.parameters())
                        
            # -------------------------- Estimate remaining time ------------------------- #
            elapsed_time = time.time() - start_time
            avg_time_per_epoch = elapsed_time / (epoch + 1)
            remaining_time = avg_time_per_epoch * (self.num_epochs - epoch - 1)
            # remaining time in hh:mm:ss format
            remaining_time_hms = time.strftime("%H:%M:%S", time.gmtime(remaining_time))
            # Estimated time that the model is expected to finish training
            estimated_finish_time = time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() + remaining_time))
            print(f"Remaining time: {remaining_time_hms}, Estimated finish time: {estimated_finish_time}")
        send_notification(
            title="Training complete",
            message=f"Training of {self.output_dir} is complete! Total training time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}")
            
        
            
            
        self.accelerator.end_training()
    
    def train_vae(self):
        
        for epoch in range(self.first_epoch, self.args.num_epochs):
            
            self.model.train()
            self.progress_bar = tqdm(total=self.num_update_steps_per_epoch)
            self.progress_bar.set_description(f"Epoch {epoch}")
            for step, batch in enumerate(self.train_dataloader):
                # Skip steps until we reach the resumed step
                if self.args.resume_from_checkpoint and epoch == self.first_epoch and step < self.resume_step:
                    if step % self.args.gradient_accumulation_steps == 0:
                        self.progress_bar.update(1)
                    continue
                
                
                class_labels = batch["class"]
                class_labels = class_labels.to(self.accelerator.device)
                clean_images = batch["image"].to(self.accelerator.device)
                
                
                # print(f"Clean images shape: {clean_images.shape}, dtype: {clean_images.dtype}")
                # VAE ENCODING:
                with torch.no_grad():
                    clean_images = self.vae.encode(clean_images).latent_dist.sample()
                    # Scale latents by VAE scaling factor (important!)
                    clean_images = clean_images * self.vae.config.scaling_factor
                    # print(f"Latent images shape: {clean_images.shape}, dtype: {clean_images.dtype}")
                    
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

                    if self.args.prediction_type == "epsilon":
                        loss = F.mse_loss(model_output.float(), noise.float())  # this could have different weights!
                    elif self.args.prediction_type == "sample":
                        alpha_t = self._extract_into_tensor(
                            self.noise_scheduler.alphas_cumprod, timesteps, (clean_images.shape[0], 1, 1, 1)
                        )
                        snr_weights = alpha_t / (1 - alpha_t)
                        # use SNR weighting from distillation paper
                        loss = snr_weights * F.mse_loss(model_output.float(), clean_images.float(), reduction="none")
                        loss = loss.mean()
                    else:
                        raise ValueError(f"Unsupported prediction type: {self.args.prediction_type}")
                    
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
            if epoch % self.args.save_images_epochs == 0 or epoch == self.args.num_epochs - 1:
                unet = self.accelerator.unwrap_model(self.model)
                images_processed = self.inference(unet,
                                                scheduler=self.noise_scheduler,
                                                vae=self.vae
                                                )

                tracker = self.accelerator.get_tracker("tensorboard", unwrap=True)
                tracker.add_images("test_samples", images_processed.transpose(0, 3, 1, 2), epoch)
            
            if epoch % self.args.save_model_epochs == 0 or epoch == self.args.num_epochs - 1:
                # save the model
                unet = self.accelerator.unwrap_model(self.model)

                if self.args.use_ema:
                    self.ema_model.store(unet.parameters())
                    self.ema_model.copy_to(unet.parameters())

                pipeline = DDIMPipeline(
                    unet=unet,
                    scheduler=self.noise_scheduler,
                )

                pipeline.save_pretrained(self.args.output_dir)

                if self.args.use_ema:
                    self.ema_model.restore(unet.parameters())
            
            
            self.accelerator.end_training()    
                
    def save_checkpoint(self, loss):
        # Checks if the accelerator has performed an optimization step behind the scenes
        if self.accelerator.sync_gradients:
            if self.args.use_ema:
                self.ema_model.step(self.model.parameters())
            self.progress_bar.update(1)
            self.global_step += 1
            if self.global_step % self.args.checkpointing_steps == 0:
                # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                if self.args.checkpoints_total_limit is not None:
                    checkpoints = os.listdir(self.args.output_dir)
                    checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                    # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                    if len(checkpoints) >= self.args.checkpoints_total_limit:
                        num_to_remove = len(checkpoints) - self.args.checkpoints_total_limit + 1
                        removing_checkpoints = checkpoints[0:num_to_remove]

                        self.logger.info(
                            f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                        )
                        self.logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                        for removing_checkpoint in removing_checkpoints:
                            removing_checkpoint = os.path.join(self.args.output_dir, removing_checkpoint)
                            shutil.rmtree(removing_checkpoint)

                save_path = os.path.join(self.args.output_dir, f"checkpoint-{self.global_step}")
                self.accelerator.save_state(save_path)
                self.logger.info(f"Saved state to {save_path}")

        logs = {"loss": loss.detach().item(), "lr": self.lr_scheduler.get_last_lr()[0], "step": self.global_step}
        if self.args.use_ema:
            logs["ema_decay"] = self.ema_model.cur_decay_value
        self.progress_bar.set_postfix(**logs)
        self.accelerator.log(logs, step=self.global_step)
        
    def inference(self, unet, scheduler=None, vae=None):
        
        if self.args.use_ema:
            self.ema_model.store(unet.parameters())
            self.ema_model.copy_to(unet.parameters())

        pipeline = Pipeline(
            unet=unet,
            scheduler=scheduler,
        )
        generator = torch.Generator(device=pipeline.device).manual_seed(0)
        # run pipeline in inference (sample random noise and denoise)
        # Get latents from pipeline

        print("Running inference with class conditioning.")
        # Create class labels for evaluation.
        class_labels = torch.from_numpy(np.linspace(0, self.num_classes - 1, self.args.eval_batch_size, dtype=int)).to(self.accelerator.device)
        
        latents = pipeline(
            generator=generator,
            batch_size=self.args.eval_batch_size,
            num_inference_steps=self.args.ddpm_num_inference_steps,
            output_type="latent",
            class_labels=class_labels,
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
            

        if self.args.use_ema:
            self.ema_model.restore(unet.parameters())
            
        images = latents
        self.logger.info(f"Generated latents shape: {images.shape}")
        
        # denormalize the images (VAE outputs are in [-1, 1] range)
        images_processed = ((images / 2 + 0.5).clip(0, 1) * 255).round().astype("uint8")
        return images_processed
    
    def display_batch(self, batch, num_images=9):
        '''
        Function for displaying a batch of images from the dataloader.
        --batch: A batch of images from the dataloader.
        --num_images: The number of images to display from the batch.
        '''
        #DONE
        images = batch["image"]
        
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
                img = images[idx].permute(1, 2, 0).numpy()
                ax.imshow(img)
                ax.axis('off')
            else:
                ax.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        
        
    


if __name__ == "__main__":
    trainer = training()
        