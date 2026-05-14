# ---------------------------------------------------------------------------- #
#                                    Imports                                   #
# ---------------------------------------------------------------------------- #
import json
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# Import resnet18 weights enum
from torchvision.models import ResNet18_Weights
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import time
import logging
logger = logging.getLogger(__name__)


from resnet_dataloader import ResNetDataloader
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path

# Import parse args
from argparse import ArgumentParser
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
try:
    from src.summarize_training.generate_config_summary import generate_data_summary_from_config
    from src.testing.evaluate_resnet18 import evaluate_resnet18
except ImportError as e:
    logger.warning(f"Could not import custom training session modules: {e}")
    logger.warning("Make sure you have the 'src/summarize_training' folder.")
    evaluate_resnet18 = None
    pass

def load_resnet18(num_classes):
    model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class ResNet18Test:
    def __init__(self, config_path, resnet_dataloader: ResNetDataloader=None, output_dir=None, resume_checkpoint_path=None, run_number=None):
        self.config_path = config_path
        logger.info(f"Loading config from: {config_path}")
        
        # Load hyperparameters and logging settings from config
        with open(config_path, "r") as f:
            self.config = json.load(f)
        self.batch_size = self.config["hyperparameters"]["batch_size"]
        self.num_epochs = self.config["hyperparameters"]["epochs"]
        self.learning_rate = self.config["hyperparameters"]["learning_rate"]
        try:
            self.warmup_epochs = int(self.config["hyperparameters"].get("warmup_epochs", 0))
        except (TypeError, ValueError):
            logger.warning("Invalid hyperparameters.warmup_epochs. Falling back to 0.")
            self.warmup_epochs = 0
        self.weight_decay = self.config["hyperparameters"].get("weight_decay", 0.0005)
        
        self.run_number = run_number
        
        self.output_dir = output_dir if output_dir is not None else self.config["logging"]["output_dir"] + "/resnet18/"
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load data first so we can extract seed from dataloader if available
        self.load_data(resnet_dataloader)
        
        # Determine seed: priority is run_number > dataloader seed > config seed
        if run_number is not None:
            self.seed = run_number
            logger.info(f"Using deterministic seed based on run_number: {self.seed}")
        elif hasattr(self.waste_dataloader, 'seed'):
            self.seed = self.waste_dataloader.seed
            logger.info(f"Using seed from dataloader: {self.seed}")
        else:
            self.seed = self.config["logging"].get("seed", 42)
            logger.info(f"Using config seed: {self.seed}")
        
        # Load model 
        self.model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
        
        # Loss function, optimizer, device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scheduler = None
        self.setup_scheduler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # Resume from checkpoint
        self.resume_from_checkpoint(resume_checkpoint_path)
        
        # Create a separate CUDA stream for data transfers to enable overlapping
        if torch.cuda.is_available():
            self.transfer_stream = torch.cuda.Stream(device=self.device)
        else:
            self.transfer_stream = None
        
        
        
    def setup_scheduler(self):
        scheduler_cfg = self.config.get("scheduler", self.config.get("hyperparameters", {}).get("scheduler", {}))
        if not isinstance(scheduler_cfg, dict):
            logger.warning("Invalid scheduler config type '%s'. Scheduler disabled.", type(scheduler_cfg).__name__)
            return

        if not scheduler_cfg.get("enabled", False):
            return

        scheduler_type = str(scheduler_cfg.get("type", "cosine")).lower()
        if scheduler_type != "cosine":
            logger.warning(
                "Only cosine scheduler is supported for ResNet18 training. Got '%s'. Scheduler disabled.",
                scheduler_type,
            )
            return

        try:
            warmup_epochs = max(0, self.warmup_epochs)
            if warmup_epochs != self.warmup_epochs:
                logger.warning("Negative warmup_epochs is not allowed. Using 0.")

            t_max = int(scheduler_cfg.get("t_max", max(1, self.num_epochs - warmup_epochs)))
            t_max = max(1, t_max)
            eta_min = float(scheduler_cfg.get("eta_min", 0.0))

            cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=t_max, eta_min=eta_min)

            if warmup_epochs > 0:
                warmup_start_factor = max(1e-6, 1.0 / float(warmup_epochs))
                warmup_scheduler = optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=warmup_start_factor,
                    end_factor=1.0,
                    total_iters=warmup_epochs,
                )
                self.scheduler = optim.lr_scheduler.SequentialLR(
                    self.optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[warmup_epochs],
                )
                logger.info(
                    "Enabled cosine scheduler with warmup (warmup_epochs=%s, t_max=%s, eta_min=%s)",
                    warmup_epochs,
                    t_max,
                    eta_min,
                )
            else:
                self.scheduler = cosine_scheduler
                logger.info("Enabled CosineAnnealingLR scheduler (t_max=%s, eta_min=%s)", t_max, eta_min)
        except (TypeError, ValueError) as e:
            logger.warning("Failed to initialize scheduler '%s': %s. Scheduler disabled.", scheduler_type, e)
            self.scheduler = None

    def _checkpoint_payload(self, epoch):
        payload = {
            "epoch": epoch,
            "seed": self.seed,
            "run_number": self.run_number,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_loss": self.log_train_loss,
            "val_loss": self.log_val_loss,
            "train_acc": self.log_train_acc,
            "val_acc": self.log_val_acc,
            "val_f1": self.log_val_f1,
        }
        if self.scheduler is not None:
            payload["scheduler_state_dict"] = self.scheduler.state_dict()
        return payload

    def resume_from_checkpoint(self, resume_checkpoint_path=None):
        self.start_epoch = 0
        self.best_val_acc = -1.0
        self.best_f1 = -1.0
        self.lowest_val_loss = float("inf")
        self.log_train_loss, self.log_val_loss, self.log_train_acc, self.log_val_acc = [], [], [], []
        self.log_val_f1 = []
        
        self.checkpointing_steps = self.config["logging"].get("checkpointing_steps", 5)
        resume_from_checkpoint = self.config["logging"].get("resume_from_checkpoint", True)
        checkpoint_dir = self.config["logging"].get("checkpoint_dir", None)
        # Metric used to choose the "best" checkpoint: 'val_acc' (default) or 'f1'
        self.checkpoint_metric = self.config["logging"].get("checkpoint_metric", "val_acc")
        
        if resume_checkpoint_path is not None:
            checkpoint_dir = resume_checkpoint_path
        logger.debug(f"Resume from checkpoint: {resume_from_checkpoint}, checkpoint dir: {checkpoint_dir}")
        if resume_from_checkpoint and checkpoint_dir is not None:
            if os.path.isfile(checkpoint_dir):
                logger.info(f"Loading checkpoint from: {checkpoint_dir}")
                checkpoint = torch.load(checkpoint_dir)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                scheduler_state = checkpoint.get("scheduler_state_dict", None)
                if self.scheduler is not None and scheduler_state is not None:
                    self.scheduler.load_state_dict(scheduler_state)
                self.start_epoch = checkpoint["epoch"] + 1
                self.log_train_loss = checkpoint["train_loss"]
                self.log_val_loss = checkpoint["val_loss"]
                self.log_train_acc = checkpoint["train_acc"]
                self.log_val_acc = checkpoint["val_acc"]
                # optional F1 log
                self.log_val_f1 = checkpoint.get("val_f1", [])
                if len(self.log_val_f1) > 0:
                    self.best_f1 = max(self.log_val_f1)
                self.best_val_acc = max(self.log_val_acc)
                self.lowest_val_loss = min(self.log_val_loss)
                logger.info(f"Resuming training from epoch {self.start_epoch}. Best val acc so far: {self.best_val_acc:.2f}%")
                
    @staticmethod
    def set_seed(seed=42):
        """Set all random seeds for reproducibility."""
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For reproducibility (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info(f"Set all random seeds to {seed}")

    def _validate_targets(self, labels: torch.Tensor, outputs: torch.Tensor, context: str) -> torch.Tensor:
        """Validate target tensor before CrossEntropyLoss to avoid opaque CUDA asserts."""
        if labels.dtype != torch.long:
            labels = labels.long()

        if outputs.ndim != 2:
            raise ValueError(
                f"Expected model outputs to have shape [N, C], got {tuple(outputs.shape)} ({context})."
            )

        n_classes = int(outputs.size(1))
        if n_classes <= 0:
            raise ValueError(f"Invalid number of classes in model output: {n_classes} ({context}).")

        if labels.numel() == 0:
            raise ValueError(f"Received empty target tensor ({context}).")

        min_label = int(labels.min().detach().cpu().item())
        max_label = int(labels.max().detach().cpu().item())
        if min_label < 0 or max_label >= n_classes:
            unique_labels = sorted(int(v) for v in torch.unique(labels.detach().cpu()).tolist())
            raise ValueError(
                "Target index out of range for CrossEntropyLoss "
                f"({context}): min={min_label}, max={max_label}, n_classes={n_classes}, "
                f"unique_labels={unique_labels}, class_names={self.class_names}"
            )

        return labels
        
    def train(self):
        # Set the random seed before training
        ResNet18Test.set_seed(self.seed)
        
        writer = SummaryWriter(self.output_dir)
        start_time = time.time()
        val_acc = 0
        val_loss = float("inf")
        
        # Timing configuration
        # Enable log_timing if logger is in debug mode or if you want detailed timing breakdowns for each epoch to identify bottlenecks. This will add some overhead due to more frequent time measurements and logging, but can be invaluable for performance tuning.
        log_timing = logger.isEnabledFor(logging.DEBUG)
                
        # Print total number of parameters in the model
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Total parameters in ResNet18: {total_params}")
        # Print total number of trainable parameters in the model
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Trainable parameters in ResNet18: {trainable_params}")

        for epoch in tqdm(range(self.start_epoch, self.num_epochs)):
            self.model.train()
            running_loss = 0.0
            train_correct, train_total = 0, 0

            # === Timing buckets for training loop ===
            time_data_load = 0.0
            time_data_transfer = 0.0
            time_batch_mixing = 0.0
            time_forward = 0.0
            time_backward = 0.0
            time_optim_step = 0.0
            time_metrics = 0.0
            
            # === GPU-side metric accumulators (avoid .item() in loop) ===
            loss_accumulator = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            correct_accumulator = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            
            end_of_step_time = time.perf_counter()
            
            for batch_idx, batch in tqdm(enumerate(self.train_loader)):
                # Time: Data loading from DataLoader
                time_data_load += time.perf_counter() - end_of_step_time
                
                labels = batch["class"]
                inputs = batch["image"]
                
                # Time: Host->Device transfer (queued on transfer stream for overlap)
                t_transfer_start = time.perf_counter()
                if self.transfer_stream is not None:
                    with torch.cuda.stream(self.transfer_stream):
                        inputs = inputs.to(self.device, non_blocking=True)
                        labels = labels.to(self.device, non_blocking=True)
                    # Ensure default stream waits for transfer completion without a full device sync.
                    torch.cuda.current_stream(self.device).wait_stream(self.transfer_stream)
                else:
                    inputs = inputs.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                time_data_transfer += time.perf_counter() - t_transfer_start
                
                # Time: Batch mixing (mixup/cutmix)
                t_mixing_start = time.perf_counter()
                mixed_inputs, labels_a, labels_b, lam, mix_mode = self.waste_dataloader.apply_batch_mixing(inputs, labels)
                time_batch_mixing += time.perf_counter() - t_mixing_start

                # Time: Forward pass
                t_forward_start = time.perf_counter()
                self.optimizer.zero_grad()
                outputs = self.model(mixed_inputs)
                time_forward += time.perf_counter() - t_forward_start

                labels_a = self._validate_targets(
                    labels_a,
                    outputs,
                    context=f"train epoch={epoch + 1} batch={batch_idx} labels_a",
                )
                labels_b = self._validate_targets(
                    labels_b,
                    outputs,
                    context=f"train epoch={epoch + 1} batch={batch_idx} labels_b",
                )
                
                # Time: Loss computation and backward pass
                t_backward_start = time.perf_counter()
                loss = self.waste_dataloader.mixed_loss(self.criterion, outputs, labels_a, labels_b, lam)
                loss.backward()
                time_backward += time.perf_counter() - t_backward_start
                
                # Time: Optimizer step
                t_optim_start = time.perf_counter()
                self.optimizer.step()
                time_optim_step += time.perf_counter() - t_optim_start
                
                # Time: Metric calculation (OPTIMIZED: GPU tensors, no sync in loop)
                t_metrics_start = time.perf_counter()
                # Accumulate loss on GPU (as tensor, no sync)
                loss_accumulator += loss.detach() * mixed_inputs.size(0)
                # Accumulate correct predictions on GPU
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                if mix_mode == "none":
                    # Keep on GPU - only convert to float tensor, no .item() call
                    correct_accumulator += (predicted.eq(labels).sum().float())
                else:
                    correct_a = predicted.eq(labels_a).sum().float()
                    correct_b = predicted.eq(labels_b).sum().float()
                    correct_accumulator += lam * correct_a + (1.0 - lam) * correct_b
                time_metrics += time.perf_counter() - t_metrics_start
                
                

                end_of_step_time = time.perf_counter()
            
            # === Single GPU-to-CPU transfer at end of epoch ===
            running_loss = loss_accumulator.item()
            train_correct = correct_accumulator.item()

            # Log timing for training loop
            num_batches = len(self.train_loader)
            if log_timing and num_batches > 0:
                logger.info(f"[TIMING] Epoch {epoch+1} training loop breakdown ({num_batches} batches):")
                logger.info(f"  - Data loading:      {time_data_load:8.2f}s ({time_data_load/num_batches:6.3f}s/batch avg)")
                logger.info(f"  - Host→Device:       {time_data_transfer:8.2f}s ({time_data_transfer/num_batches:6.3f}s/batch avg)")
                logger.info(f"  - Batch mixing:      {time_batch_mixing:8.2f}s ({time_batch_mixing/num_batches:6.3f}s/batch avg)")
                logger.info(f"  - Forward pass:      {time_forward:8.2f}s ({time_forward/num_batches:6.3f}s/batch avg)")
                logger.info(f"  - Loss + Backward:   {time_backward:8.2f}s ({time_backward/num_batches:6.3f}s/batch avg)")
                logger.info(f"  - Optimizer step:    {time_optim_step:8.2f}s ({time_optim_step/num_batches:6.3f}s/batch avg)")
                logger.info(f"  - Metric tracking:   {time_metrics:8.2f}s ({time_metrics/num_batches:6.3f}s/batch avg)")
                total_train_time = time_data_load + time_data_transfer + time_batch_mixing + time_forward + time_backward + time_optim_step + time_metrics
                logger.info(f"  - TOTAL:             {total_train_time:8.2f}s")
        
            train_loss = running_loss / len(self.train_loader.dataset)
            train_acc = 100. * train_correct / train_total
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Accuracy/train", train_acc, epoch)
            
            # === Validation loop with timing ===
            t_val_start = time.perf_counter()
            self.model.eval()
            val_loss_sum = 0.0
            val_correct, val_total = 0, 0
            preds_list = []
            labels_list = []
            with torch.no_grad():
                for batch in self.val_loader:
                    labels = batch["class"]
                    inputs = batch["image"]
                    inputs = inputs.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    
                    outputs = self.model(inputs)
                    labels = self._validate_targets(
                        labels,
                        outputs,
                        context=f"val epoch={epoch + 1}",
                    )
                    loss = self.criterion(outputs, labels)
                    val_loss_sum += loss.item() * inputs.size(0)
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    preds_list.append(predicted.cpu().numpy())
                    labels_list.append(labels.cpu().numpy())
            val_loss = val_loss_sum / len(self.val_loader.dataset)
            val_acc = 100.0 * val_correct / val_total
            time_val = time.perf_counter() - t_val_start
            if log_timing:
                logger.info(f"[TIMING] Epoch {epoch+1} validation: {time_val:.2f}s ({len(self.val_loader)} batches)")

            # Compute macro F1 (multi-class) - vectorized
            if len(preds_list) > 0:
                preds_all = np.concatenate(preds_list)
                labels_all = np.concatenate(labels_list)
                num_classes = self.num_classes
                eps = 1e-8
                tp = np.zeros(num_classes, dtype=np.int64)
                pred_counts = np.zeros(num_classes, dtype=np.int64)
                true_counts = np.zeros(num_classes, dtype=np.int64)
                for c in range(num_classes):
                    tp[c] = int(((preds_all == c) & (labels_all == c)).sum())
                    pred_counts[c] = int((preds_all == c).sum())
                    true_counts[c] = int((labels_all == c).sum())
                precision = tp / (pred_counts + eps)
                recall = tp / (true_counts + eps)
                f1_per_class = 2 * precision * recall / (precision + recall + eps)
                val_f1 = float(np.mean(f1_per_class))
            else:
                val_f1 = 0.0

            self.log_val_loss.append(val_loss)
            self.log_val_acc.append(val_acc)
            self.log_val_f1.append(val_f1)
            self.log_train_loss.append(train_loss)
            self.log_train_acc.append(train_acc)

            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Accuracy/val", val_acc, epoch)
            writer.add_scalar("F1/val", val_f1, epoch)

            if self.scheduler is not None:
                prev_lr = float(self.optimizer.param_groups[0]["lr"])
                self.scheduler.step()
                new_lr = float(self.optimizer.param_groups[0]["lr"])
                if new_lr != prev_lr:
                    logger.info("LR scheduler update: %.8f -> %.8f", prev_lr, new_lr)

            writer.add_scalar("LearningRate", float(self.optimizer.param_groups[0]["lr"]), epoch)

            # ----- Save checkpoint ----- #
            t_ckpt_start = time.perf_counter()
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                output_name = f"resnet18_best_val_acc.ckpt"
                output_checkpoint_path = os.path.join(self.output_dir, output_name)
                torch.save(self._checkpoint_payload(epoch), output_checkpoint_path)
                logging.info(f"New best val acc: {self.best_val_acc:.2f}%. Model checkpoint saved: {output_checkpoint_path}")
            # Optionally save checkpoint based on highest F1 instead of val_acc
            if getattr(self, "checkpoint_metric", "val_acc") == "f1":
                # val_f1 should be defined for this epoch
                try:
                    if val_f1 > self.best_f1:
                        self.best_f1 = val_f1
                        output_name = f"resnet18_best_val_f1.ckpt"
                        output_checkpoint_path = os.path.join(self.output_dir, output_name)
                        torch.save(self._checkpoint_payload(epoch), output_checkpoint_path)
                        logging.info(f"New best val F1: {self.best_f1:.4f}. Model checkpoint saved: {output_checkpoint_path}")
                except NameError:
                    pass
                
            if val_loss < self.lowest_val_loss:
                self.lowest_val_loss = val_loss
                output_name = f"resnet18_lowest_val_loss.ckpt"
                output_checkpoint_path = os.path.join(self.output_dir, output_name)
                torch.save(self._checkpoint_payload(epoch), output_checkpoint_path)
                logging.info(f"New lowest val loss: {self.lowest_val_loss:.4f}. Model checkpoint saved: {output_checkpoint_path}")
                
                
            # Save checkpoint
            if (epoch + 1) % self.checkpointing_steps == 0 or epoch == self.num_epochs - 1:
                output_name = f"resnet18_latest.ckpt"
                output_checkpoint_path = os.path.join(self.output_dir, output_name)
                torch.save(self._checkpoint_payload(epoch), output_checkpoint_path)
                # Update config file with checkpoint directory
                self.config["logging"]["checkpoint_dir"] = str(output_checkpoint_path)
                logging.info(f"Checkpoint saved: {output_checkpoint_path}")
                logging.debug(f"Config path: {self.config_output_path}")
                with open(self.config_output_path, "w") as f:
                    json.dump(self.config, f)
            time_ckpt = time.perf_counter() - t_ckpt_start
            if log_timing:
                logger.info(f"[TIMING] Epoch {epoch+1} checkpoint save: {time_ckpt:.2f}s")
            
            # -------------------------- Estimate remaining time ------------------------- #
            elapsed_time = time.time() - start_time
            # When resuming from a checkpoint, use epochs completed in this run.
            completed_epochs = (epoch - self.start_epoch + 1)
            avg_time_per_epoch = elapsed_time / max(1, completed_epochs)
            remaining_time = avg_time_per_epoch * (self.num_epochs - epoch - 1)
            # remaining time in hh:mm:ss format
            remaining_time_hms = time.strftime("%H:%M:%S", time.gmtime(remaining_time))
            # Estimated time that the model is expected to finish training
            estimated_finish_time = time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() + remaining_time))
            tqdm.write(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Remaining time: {remaining_time_hms}, Estimated finish time: {estimated_finish_time}")
        logger.info("Training complete.")
        # Rename latest checkpoint to include final val acc and val loss
        final_checkpoint_path = os.path.join(self.output_dir, f"resnet18_final_valacc{val_acc:.2f}_valloss{val_loss:.4f}.ckpt")
        os.rename(os.path.join(self.output_dir, "resnet18_latest.ckpt"), final_checkpoint_path)
        writer.close()
        
    def load_resnet18(self, num_classes):
        model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
        
    def load_data(self, resnet_dataloader: ResNetDataloader = None):
        self.waste_dataloader = resnet_dataloader if resnet_dataloader is not None else ResNetDataloader(self.config_path)
        self.train_loader = self.waste_dataloader.train_loader
        self.val_loader = self.waste_dataloader.val_loader
        self.class_names = self.waste_dataloader.classes
        self.num_classes = self.waste_dataloader.num_classes
        self.config_output_path = self.waste_dataloader.output_config_path

        

if __name__ == "__main__":
    # Load config
    parser = ArgumentParser()
    parser.add_argument("--config", default="src/testing/testing_config.json", help="Path to config file")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation on test/validation set after training")
    # arg = ["--config", "src/testing/testing_config.json", "--evaluate"]
    args = parser.parse_args()
    config_path = args.config
    resnet_trainer = ResNet18Test(config_path)
    resnet_trainer.train()
    
    # Run evaluation if requested
    if args.evaluate and evaluate_resnet18 is not None:
        logger.info("\n" + "="*80)
        logger.info("Running post-training evaluation...")
        logger.info("="*80 + "\n")
        try:
            # Use the best validation accuracy checkpoint
            best_checkpoint = os.path.join(resnet_trainer.output_dir, "resnet18_best_val_acc.ckpt")
            if os.path.exists(best_checkpoint):
                evaluate_resnet18(dataloader=resnet_trainer.waste_dataloader, checkpoint_dir=resnet_trainer.output_dir, best_checkpoint="lowest_val_loss")
            else:
                logger.warning("Best checkpoint not found, skipping evaluation")
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    elif args.evaluate:
        logger.warning("Evaluation not available, evaluate_resnet18 could not be imported")

    
    
    
    