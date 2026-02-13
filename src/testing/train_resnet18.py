# ---------------------------------------------------------------------------- #
#                                    Imports                                   #
# ---------------------------------------------------------------------------- #
import json
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



def load_resnet18(num_classes):
    model = torchvision.models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model




if __name__ == "__main__":
    # Load config
    config_path = "/home/ap/cloud/Master/aris_master/testing/testing_dir/config.json"
    print(f"Loading config from: {config_path}")
    
    
    # Extract config parameters
    with open(config_path, "r") as f:
        config = json.load(f)
    batch_size = config["hyperparameters"]["batch_size"]
    num_epochs = config["hyperparameters"]["epochs"]
    learning_rate = config["hyperparameters"]["learning_rate"]
    
    
    weight_decay = 0.0005
    
    output_dir = config["logging"]["output_dir"] + "/resnet18/"
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    
    # Load data
    waste_dataloader = ResNetDataloader(config_path)
    train_loader = waste_dataloader.train_loader
    val_loader = waste_dataloader.val_loader
    class_names = waste_dataloader.classes
    num_classes = waste_dataloader.num_classes
    config_output_path = waste_dataloader.output_config_path



    # Load model 
    model = load_resnet18(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Check if the model should continue training from a checkpoint
    start_epoch = 0
    best_val_acc = -1.0
    lowest_val_loss = float("inf")
    checkpointing_steps = config["logging"]["checkpointing_steps"]
    resume_from_checkpoint = config["logging"]["resume_from_checkpoint"]
    checkpoint_dir = config["logging"]["checkpoint_dir"]
    if resume_from_checkpoint and checkpoint_dir is not None:
        if os.path.isfile(checkpoint_dir):
            print(f"Loading checkpoint from: {checkpoint_dir}")
            checkpoint = torch.load(checkpoint_dir)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            log_train_loss = checkpoint["train_loss"]
            log_val_loss = checkpoint["val_loss"]
            log_train_acc = checkpoint["train_acc"]
            log_val_acc = checkpoint["val_acc"]
            best_val_acc = max(log_val_acc)
            lowest_val_loss = min(log_val_loss)
            print(f"Resuming training from epoch {start_epoch}. Best val acc so far: {best_val_acc:.2f}%")

    
    
    # Training loop
    writer = SummaryWriter(output_dir)
    start_time = time.time()

    log_train_loss, log_val_loss, log_train_acc, log_val_acc = [], [], [], []
    for epoch in tqdm(range(start_epoch, num_epochs)):
        model.train()
        running_loss = 0.0
        train_correct, train_total = 0, 0
        for batch_idx, batch in enumerate(train_loader):
            labels = batch["class"]
            inputs = batch["image"]
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Track training accuracy
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            # Log batch loss
            global_step = epoch * len(train_loader) + batch_idx
    
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = 100. * train_correct / train_total
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        
        model.eval()
        val_loss_sum = 0.0
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                labels = batch["class"]
                inputs = batch["image"]
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss_sum += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        val_loss = val_loss_sum / len(val_loader.dataset)
        val_acc = 100.0 * val_correct / val_total
        
        log_val_loss.append(val_loss)
        log_val_acc.append(val_acc)
        log_train_loss.append(train_loss)
        log_train_acc.append(train_acc)

        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        # ----- Save checkpoint ----- #
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            #output_name = f"resnet18_epoch{epoch+1}_valacc{val_acc:.2f}_val_loss{val_loss:.4f}.ckpt"
            output_name = f"resnet18_best_val_acc.ckpt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": log_train_loss,
                "val_loss": log_val_loss,
                "train_acc": log_train_acc,
                "val_acc": log_val_acc,
            }, os.path.join(output_dir, output_name))
            print(f"New best val acc: {best_val_acc:.2f}%. Model checkpoint saved: {output_name}")
            
        if val_loss < lowest_val_loss:
            lowest_val_loss = val_loss
            #output_name = f"resnet18_epoch{epoch+1}_valacc{val_acc:.2f}_val_loss{val_loss:.4f}.ckpt"
            output_name = f"resnet18_lowest_val_loss.ckpt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": log_train_loss,
                "val_loss": log_val_loss,
                "train_acc": log_train_acc,
                "val_acc": log_val_acc,
            }, os.path.join(output_dir, output_name))
            print(f"New lowest val loss: {lowest_val_loss:.4f}. Model checkpoint saved: {output_name}")
            
            
        # Save checkpoint
        if (epoch + 1) % checkpointing_steps == 0:
            output_name = f"resnet18_epoch{epoch+1}_valacc{val_acc:.2f}_val_loss{val_loss:.4f}.ckpt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": log_train_loss,
                "val_loss": log_val_loss,
                "train_acc": log_train_acc,
                "val_acc": log_val_acc,
            }, os.path.join(output_dir, output_name))
            # Update config file with checkpoint directory
            config["logging"]["checkpoint_dir"] = str(os.path.join(output_dir, output_name))
            with open(config_output_path, "w") as f:
                json.dump(config, f)
            print(f"Checkpoint saved: {output_name}")
        
        
        # -------------------------- Estimate remaining time ------------------------- #
        elapsed_time = time.time() - start_time
        avg_time_per_epoch = elapsed_time / (epoch + 1)
        remaining_time = avg_time_per_epoch * (num_epochs - epoch - 1)
        # Remaining time in minutes
        remaining_time_minutes = remaining_time / 60
        # remaining time in hh:mm:ss format
        remaining_time_hms = time.strftime("%H:%M:%S", time.gmtime(remaining_time))
        # Estimated time that the model is expected to finish training
        estimated_finish_time = time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() + remaining_time))
        tqdm.write(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Remaining time: {remaining_time_hms}, Estimated finish time: {estimated_finish_time}")
    writer.close()
        
# Save checkpoint
output_name = f"resnet18_epoch{epoch+1}_valacc{val_acc:.2f}_val_loss{val_loss:.4f}.ckpt"
torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "train_loss": log_train_loss,
    "val_loss": log_val_loss,
    "train_acc": log_train_acc,
    "val_acc": log_val_acc,
    }, os.path.join(output_dir, output_name))
# Update config file with checkpoint directory
config["logging"]["checkpoint_dir"] = str(os.path.join(output_dir, output_name))
with open(config_output_path, "w") as f:
    json.dump(config, f)
print(f"Checkpoint saved: {output_name}")
    
    
    
    
    