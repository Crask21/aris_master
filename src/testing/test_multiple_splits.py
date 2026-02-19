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
from multi_run_evaluation import multi_run_evaluation

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
    splits = list(zip(real_image_counts, synthetic_image_counts))
    
    training_runs_per_split = config["data"]["training_runs_per_split"]

    # Make 'runs' folder
    runs_dir = Path(config["logging"]["output_dir"]) / "resnet18_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)    
    
    # Resume run
    # Check runs_dir for existing runs and determine the last completed run
    existing_runs = sorted(runs_dir.glob("*-real_*synthetic_run*"))
    # Extract current split and run from the last completed run
    if existing_runs:
        last_run = existing_runs[-1]
        last_run_name = last_run.name
        print(f"[INFO] Found existing runs. Last completed run: {last_run_name}")
        try:
            last_real_count = int(last_run_name.split("-real_")[0])
            last_synthetic_count = int(last_run_name.split("-synthetic_run")[0].split("_")[-1])
            last_run_number = int(last_run_name.split("-synthetic_run")[-1])
            print(f"[INFO] Last completed split: {last_real_count} real, {last_synthetic_count} synthetic, run {last_run_number}")
            # Determine the index of the last completed split
            last_split_idx = splits.index((last_real_count, last_synthetic_count))
            # Set the starting point for the next run
            start_split_idx = last_split_idx
            start_run_number = last_run_number
            
            checkpoint_dir = config["logging"]["checkpoint_dir"]
            # Check if checkpoint_dir is set to the last completed run's checkpoint
            expected_checkpoint_dir = f"{runs_dir}/{last_real_count}-real_{last_synthetic_count}-synthetic_run{last_run_number}/resnet18_latest.ckpt"
            # Check if file exists
            if not Path(expected_checkpoint_dir).exists():
                print(f"[WARNING] Expected checkpoint directory does not exist: {expected_checkpoint_dir}")
            if checkpoint_dir != expected_checkpoint_dir:
                print(f"[WARNING] Checkpoint directory in config does not match expected checkpoint for last completed run. Updating checkpoint directory to: {expected_checkpoint_dir}")
                print(f"[INFO] Config checkpoint directory: {checkpoint_dir}")
                print(f"[INFO] Expected checkpoint directory: {expected_checkpoint_dir}")
                config["logging"]["checkpoint_dir"] = None  # Set to None to avoid loading from an old checkpoint
        except Exception as e:
            print(f"[ERROR] Could not parse last run name: {e}")
            start_split_idx = 0
            start_run_number = 0
    # Start time
    start_time = time.time()
    total_runs = len(splits) * training_runs_per_split
    split_idx = 0
    for real_count, synthetic_count in splits:
        for run in range(training_runs_per_split):
            if args.train:
                print(f"\n[INFO] Training model on {real_count} real images and {synthetic_count} synthetic images...")
                # Create dataloader for this split
                dataloader = ResNetDataloader(config_path, real_image_count=real_count, synthetic_image_count=synthetic_count)  
                
                checkpoint_dir = f"{runs_dir}/{real_count}-real_{synthetic_count}-synthetic_run{run+1}/"

                # Train model on this split
                model = ResNet18Test(config_path, resnet_dataloader=dataloader, output_dir=checkpoint_dir)
                model.train()
            
            # Evaluate model trained on this split
            if args.evaluate:
                print(f"\n[INFO] Evaluating model trained on {real_count} real images and {synthetic_count} synthetic images...")
                dataloader = ResNetDataloader(config_path, real_image_count=real_count, synthetic_image_count=synthetic_count)  
                
                checkpoint_dir = f"{runs_dir}/{real_count}-real_{synthetic_count}-synthetic_run{run+1}/"
                evaluate_resnet18(dataloader, checkpoint_dir=checkpoint_dir, best_checkpoint="lowest_val_loss")
            
            # -------------------------- Estimate remaining time ------------------------- #
            elapsed_time = time.time() - start_time
            avg_time_per_epoch = elapsed_time / (split_idx*training_runs_per_split + run + 1)
            remaining_time = avg_time_per_epoch * (total_runs - (len(splits) * training_runs_per_split - split_idx * training_runs_per_split - run - 1))
            print(f"Elapsed time: {elapsed_time:.2f}s, Average time per run: {avg_time_per_epoch:.2f}s")
            # remaining time in hh:mm:ss format
            remaining_time_hms = time.strftime("%H:%M:%S", time.gmtime(remaining_time))
            # Estimated time that the model is expected to finish training
            estimated_finish_time = time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() + remaining_time))
            print(f"[INFO] Estimated remaining time: {remaining_time_hms} (Estimated finish time: {estimated_finish_time})")
            # -------------------------------------------------------------------------- #
    multi_run_evaluation(resnet18_runs_dir=str(runs_dir))
    send_notification("Multiple splits training and evaluation complete!", "All models have been trained and evaluated on their respective splits.", send_to_casper=False)