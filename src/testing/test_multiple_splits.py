import json
import sys
import time
import os
import torch
import torch.nn as nn
import torchvision
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_recall_fscore_support
)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.pushover import send_notification
from evaluate_resnet18 import evaluate_resnet18
from resnet_dataloader import ResNetDataloader
from train_resnet18 import ResNet18Test


if __name__ == "__main__":
    parser = ArgumentParser(description="Train multiple resnet18 models on different splits of the dataset and evaluate their performance.")
    parser.add_argument(
        "--config",
        type=str,
        default="/home/ap/cloud/Master/aris_master/src/testing/testing_config.json",
        #required=True,
        help="Path to config JSON file"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train models for each split"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate models for each split"
    )
    # arg = ["--config", "/home/ap/cloud/Master/aris_master/src/testing/testing_config.json", "--evaluate"]
    args = parser.parse_args()
    config_path = args.config
    
    # Load config file
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load splits from config file
    real_image_counts = config["data"]["splits"]["real_image_counts"]
    synthetic_image_counts = config["data"]["splits"]["synthetic_image_counts"]
    # Combine into a splits tuble
    splits = list(zip(real_image_counts, synthetic_image_counts))
    
    # Training runs for each split
    training_runs_per_split = config["data"]["training_runs_per_split"]
    
    # Start time
    start_time = time.time()
    total_runs = len(splits) * training_runs_per_split
    for real_count, synthetic_count in splits:
        for run in range(training_runs_per_split):
            if args.train:
                print(f"\n[INFO] Training model on {real_count} real images and {synthetic_count} synthetic images...")
                # Create dataloader for this split
                dataloader = ResNetDataloader(config_path, real_image_count=real_count, synthetic_image_count=synthetic_count)  
                
                checkpoint_dir = f"{config['logging']['output_dir']}/resnet18_{real_count}-real_{synthetic_count}-synthetic_run{run}/"
                
                # Train model on this split
                model = ResNet18Test(config_path, resnet_dataloader=dataloader, output_dir=checkpoint_dir)
                model.train()
            
            # Evaluate model trained on this split
            if args.evaluate:
                print(f"\n[INFO] Evaluating model trained on {real_count} real images and {synthetic_count} synthetic images...")
                dataloader = ResNetDataloader(config_path, real_image_count=real_count, synthetic_image_count=synthetic_count)  
                
                checkpoint_dir = f"{config['logging']['output_dir']}/resnet18_{real_count}-real_{synthetic_count}-synthetic_run{run}/"
                evaluate_resnet18(dataloader, checkpoint_dir=checkpoint_dir, best_checkpoint="lowest_val_loss")
            
            # -------------------------- Estimate remaining time ------------------------- #
            elapsed_time = time.time() - start_time
            avg_time_per_epoch = elapsed_time / (total_runs + 1)
            remaining_time = avg_time_per_epoch * (total_runs - (len(splits) * run + 1))
            # remaining time in hh:mm:ss format
            remaining_time_hms = time.strftime("%H:%M:%S", time.gmtime(remaining_time))
            # Estimated time that the model is expected to finish training
            estimated_finish_time = time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() + remaining_time))
            print(f"[INFO] Estimated remaining time: {remaining_time_hms} (Estimated finish time: {estimated_finish_time})")
            # -------------------------------------------------------------------------- #
    send_notification("Multiple splits training and evaluation complete!", "All models have been trained and evaluated on their respective splits.", send_to_casper=False)