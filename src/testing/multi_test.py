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

def resume_from_checkpoint(config, runs_dir):
    # Check runs_dir for existing runs and determine the last completed run
    existing_runs = sorted(runs_dir.glob("*-real_*synthetic_run*"))
    # Extract current split and run from the last completed run
    if existing_runs:
        last_run = existing_runs[-1]
        # Check if last run has a "resnet18_latest.ckpt" file to confirm it was completed
        if not (last_run / "resnet18_latest.ckpt").exists():
            # Check that other *.ckpt files exist 
            ckpt_files = list(last_run.glob("*.ckpt"))
            if ckpt_files:
                print(f"[WARNING] Last run was completed as no 'resnet18_latest.ckpt' file found, but other checkpoint files exist. Assuming last run was completed.")
            skip_run = True
        else:            
            skip_run = False
            print("[INFO] Found 'resnet18_latest.ckpt' file in last run. Assuming last run was completed and will be resumed.")
        
        
        last_run_name = last_run.name
        print(f"[INFO] Found existing runs. Last run: {last_run_name}")
        try:
            last_real_count = int(last_run_name.split("-real_")[0])
            last_synthetic_count = int(last_run_name.split("-synthetic_run")[0].split("_")[-1])
            last_run_number = int(last_run_name.split("-synthetic_run")[-1])
            print(f"[INFO] Last split: {last_real_count} real, {last_synthetic_count} synthetic, run {last_run_number}")
            # Determine the index of the last completed split
            last_split_idx = splits.index((last_real_count, last_synthetic_count))
            
            # Resume from the interrupted run
            print(skip_run)
            
            if skip_run:
                print(start_run_number)
                print(last_split_idx)
                start_run_number = (last_run_number + 1) % training_runs_per_split
                print(start_run_number)
                print((last_run_number ) % training_runs_per_split)
                start_split_idx = last_split_idx
                if start_run_number == 0:
                    start_run_number = 1
                    start_split_idx = last_split_idx + 1
            else:
                start_split_idx = last_split_idx
                start_run_number = last_run_number
            print(start_split_idx)
            if start_split_idx >= len(splits):
                print(f"[INFO] All splits already completed. Starting from the beginning.")
                print("Do you want to start from the beginning? (y/n): ")
                input_str = input()
                if input_str.lower() != "y":
                    print(f"[INFO] Exiting program.")
                    sys.exit(0)
                start_split_idx = 0
                start_run_number = 1
            print(f"[INFO] Resuming interrupted run: split {start_split_idx+1}/{len(splits)}, run {start_run_number}/{training_runs_per_split}")
            
        except Exception as e:
            print(f"[ERROR] Could not parse last run name: {e}")
            start_split_idx = 0
            start_run_number = 1
    else:
        print(f"[INFO] No existing runs found. Starting from beginning.")
    return start_split_idx, start_run_number

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
    
    # Initialize start indices
    start_split_idx = 0
    start_run_number = 1  # runs are 1-indexed
    
    start_split_idx, start_run_number = resume_from_checkpoint(config, runs_dir)
    # Resume run
    
    
    # Start time
    start_time = time.time()
    total_runs = len(splits) * training_runs_per_split
    
    for split_idx, (real_count, synthetic_count) in enumerate(splits):
        # Skip to the starting split
        if split_idx < start_split_idx:
            continue
            
        for run in range(start_run_number, training_runs_per_split):  # runs are 0-indexed
            # Skip already completed runs
            if split_idx == start_split_idx and run < start_run_number:
                continue
            
            print(f"\n[INFO] Processing split {split_idx+1}/{len(splits)}, run {run}/{training_runs_per_split}")
            
            if args.train:
                print(f"[INFO] Training model on {real_count} real images and {synthetic_count} synthetic images...")
                # Create dataloader for this split
                dataloader = ResNetDataloader(config_path, real_image_count=real_count, synthetic_image_count=synthetic_count)  
                
                checkpoint_dir = f"{runs_dir}/{real_count}-real_{synthetic_count}-synthetic_run{run}/"

                # Train model on this split
                model = ResNet18Test(config_path, resnet_dataloader=dataloader, output_dir=checkpoint_dir)
                model.train()
            
            # Evaluate model trained on this split
            if args.evaluate:
                print(f"[INFO] Evaluating model trained on {real_count} real images and {synthetic_count} synthetic images...")
                dataloader = ResNetDataloader(config_path, real_image_count=real_count, synthetic_image_count=synthetic_count)  
                
                checkpoint_dir = f"{runs_dir}/{real_count}-real_{synthetic_count}-synthetic_run{run}/"
                evaluate_resnet18(dataloader, checkpoint_dir=checkpoint_dir, best_checkpoint="lowest_val_loss")
            
            # -------------------------- Estimate remaining time ------------------------- #
            elapsed_time = time.time() - start_time
            completed_runs = split_idx * training_runs_per_split + run
            avg_time_per_run = elapsed_time / completed_runs
            remaining_runs = total_runs - completed_runs
            remaining_time = avg_time_per_run * remaining_runs
            print(f"[INFO] Completed {completed_runs}/{total_runs} runs. Elapsed time: {elapsed_time:.2f}s, Average time per run: {avg_time_per_run:.2f}s")
            # remaining time in hh:mm:ss format
            remaining_time_hms = time.strftime("%H:%M:%S", time.gmtime(remaining_time))
            # Estimated time that the model is expected to finish training
            estimated_finish_time = time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() + remaining_time))
            print(f"[INFO] Estimated remaining time: {remaining_time_hms} (Estimated finish time: {estimated_finish_time})")
            # -------------------------------------------------------------------------- #
    
    print(f"\n[INFO] All training and evaluation complete!")
    multi_run_evaluation(resnet18_runs_dir=str(runs_dir))