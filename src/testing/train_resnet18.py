# ---------------------------------------------------------------------------- #
#                                    Imports                                   #
# ---------------------------------------------------------------------------- #
import json
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import time


from resnet_dataloader import ResNetDataloader
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path

# Import parse args
from argparse import ArgumentParser
# --------------------------------- AP NOTES --------------------------------- #
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
try:
    from src.summarize_training.generate_config_summary import generate_data_summary_from_config
    from src.testing.evaluate_resnet18 import evaluate_resnet18
except ImportError as e:
    print(f"Could not import custom training session modules: {e}", file=sys.stderr)
    print("Make sure you have the 'src/summarize_training' folder.", file=sys.stderr)
    evaluate_resnet18 = None
    pass
# ------------------------------- AP NOTES END ------------------------------- #

def load_resnet18(num_classes):
    model = torchvision.models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class ResNet18Test:
    def __init__(self, config_path):
        self.config_path = config_path
        print(f"Loading config from: {config_path}")
        
        # Load hyperparameters and logging settings from config
        with open(config_path, "r") as f:
            self.config = json.load(f)
        self.batch_size = self.config["hyperparameters"]["batch_size"]
        self.num_epochs = self.config["hyperparameters"]["epochs"]
        self.learning_rate = self.config["hyperparameters"]["learning_rate"]
        self.weight_decay = 0.0005
        
        self.output_dir = self.config["logging"]["output_dir"] + "/resnet18/"
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load data
        self.load_data()
        
        # Load model 
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
        
        # Loss function, optimizer, device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # Resume from checkpoint
        self.resume_from_checkpoint()
        
        
        
    def resume_from_checkpoint(self):
        self.start_epoch = 0
        self.best_val_acc = -1.0
        self.lowest_val_loss = float("inf")
        self.log_train_loss, self.log_val_loss, self.log_train_acc, self.log_val_acc = [], [], [], []
        
        self.checkpointing_steps = self.config["logging"]["checkpointing_steps"]
        resume_from_checkpoint = self.config["logging"]["resume_from_checkpoint"]
        checkpoint_dir = self.config["logging"]["checkpoint_dir"]
        if resume_from_checkpoint and checkpoint_dir is not None:
            if os.path.isfile(checkpoint_dir):
                print(f"Loading checkpoint from: {checkpoint_dir}")
                checkpoint = torch.load(checkpoint_dir)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.start_epoch = checkpoint["epoch"] + 1
                self.log_train_loss = checkpoint["train_loss"]
                self.log_val_loss = checkpoint["val_loss"]
                self.log_train_acc = checkpoint["train_acc"]
                self.log_val_acc = checkpoint["val_acc"]
                self.best_val_acc = max(self.log_val_acc)
                self.lowest_val_loss = min(self.log_val_loss)
                print(f"Resuming training from epoch {self.start_epoch}. Best val acc so far: {self.best_val_acc:.2f}%")
        
    def train(self):
        writer = SummaryWriter(self.output_dir)
        start_time = time.time()

        for epoch in tqdm(range(self.start_epoch, self.num_epochs)):
            self.model.train()
            running_loss = 0.0
            train_correct, train_total = 0, 0
            for batch_idx, batch in tqdm(enumerate(self.train_loader)):
                labels = batch["class"]
                inputs = batch["image"]
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                # Track training accuracy
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
        
            train_loss = running_loss / len(self.train_loader.dataset)
            train_acc = 100. * train_correct / train_total
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Accuracy/train", train_acc, epoch)
            
            self.model.eval()
            val_loss_sum = 0.0
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for batch in self.val_loader:
                    labels = batch["class"]
                    inputs = batch["image"]
                    inputs = inputs.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    val_loss_sum += loss.item() * inputs.size(0)
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            val_loss = val_loss_sum / len(self.val_loader.dataset)
            val_acc = 100.0 * val_correct / val_total
            
            self.log_val_loss.append(val_loss)
            self.log_val_acc.append(val_acc)
            self.log_train_loss.append(train_loss)
            self.log_train_acc.append(train_acc)

            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Accuracy/val", val_acc, epoch)

            # ----- Save checkpoint ----- #
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                #output_name = f"resnet18_epoch{epoch+1}_valacc{val_acc:.2f}_val_loss{val_loss:.4f}.ckpt"
                output_name = f"resnet18_best_val_acc.ckpt"
                output_checkpoint_path = os.path.join(self.output_dir, output_name)
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "train_loss": self.log_train_loss,
                    "val_loss": self.log_val_loss,
                    "train_acc": self.log_train_acc,
                    "val_acc": self.log_val_acc,
                }, output_checkpoint_path)
                print(f"New best val acc: {self.best_val_acc:.2f}%. Model checkpoint saved: {output_checkpoint_path}")
                
            if val_loss < self.lowest_val_loss:
                self.lowest_val_loss = val_loss
                #output_name = f"resnet18_epoch{epoch+1}_valacc{val_acc:.2f}_val_loss{val_loss:.4f}.ckpt"
                output_name = f"resnet18_lowest_val_loss.ckpt"
                output_checkpoint_path = os.path.join(self.output_dir, output_name)
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "train_loss": self.log_train_loss,
                    "val_loss": self.log_val_loss,
                    "train_acc": self.log_train_acc,
                    "val_acc": self.log_val_acc,
                }, output_checkpoint_path)
                print(f"New lowest val loss: {self.lowest_val_loss:.4f}. Model checkpoint saved: {output_checkpoint_path}")
                
                
            # Save checkpoint
            if (epoch + 1) % self.checkpointing_steps == 0:
                output_name = f"resnet18_epoch{epoch+1}_valacc{val_acc:.2f}_val_loss{val_loss:.4f}.ckpt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "train_loss": self.log_train_loss,
                    "val_loss": self.log_val_loss,
                    "train_acc": self.log_train_acc,
                    "val_acc": self.log_val_acc,
                }, os.path.join(self.output_dir, output_name))
                # Update config file with checkpoint directory
                self.config["logging"]["checkpoint_dir"] = str(os.path.join(self.output_dir, output_name))
                with open(self.config_output_path, "w") as f:
                    json.dump(self.config, f)
                print(f"Checkpoint saved: {output_name}")
            
            
            # -------------------------- Estimate remaining time ------------------------- #
            elapsed_time = time.time() - start_time
            avg_time_per_epoch = elapsed_time / (epoch + 1)
            remaining_time = avg_time_per_epoch * (self.num_epochs - epoch - 1)
            # Remaining time in minutes
            remaining_time_minutes = remaining_time / 60
            # remaining time in hh:mm:ss format
            remaining_time_hms = time.strftime("%H:%M:%S", time.gmtime(remaining_time))
            # Estimated time that the model is expected to finish training
            estimated_finish_time = time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() + remaining_time))
            tqdm.write(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Remaining time: {remaining_time_hms}, Estimated finish time: {estimated_finish_time}")
        print("Training complete.")
        writer.close()
        
    def load_resnet18(self, num_classes):
        model = torchvision.models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
        
    def load_data(self):
        self.waste_dataloader = ResNetDataloader(self.config_path)
        self.train_loader = self.waste_dataloader.train_loader
        self.val_loader = self.waste_dataloader.val_loader
        self.class_names = self.waste_dataloader.classes
        self.num_classes = self.waste_dataloader.num_classes
        self.config_output_path = self.waste_dataloader.output_config_path

        

if __name__ == "__main__":
    # Load config
    parser = ArgumentParser()
    parser.add_argument("--config", default="testing/testing_dir/config.json", help="Path to config file")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation on test/validation set after training")
    args = parser.parse_args()
    config_path = args.config
    resnet_trainer = ResNet18Test(config_path)
    resnet_trainer.train()
    
    # Run evaluation if requested
    if args.evaluate and evaluate_resnet18 is not None:
        print("\n" + "="*80)
        print("Running post-training evaluation...")
        print("="*80 + "\n")
        try:
            # Use the best validation accuracy checkpoint
            best_checkpoint = os.path.join(resnet_trainer.output_dir, "resnet18_best_val_acc.ckpt")
            if os.path.exists(best_checkpoint):
                evaluate_resnet18(config_path=config_path, checkpoint_path=best_checkpoint)
            else:
                print("[WARNING] Best checkpoint not found, skipping evaluation")
        except Exception as e:
            print(f"[ERROR] Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
    elif args.evaluate:
        print("[WARNING] Evaluation not available, evaluate_resnet18 could not be imported")

    
    
    
    