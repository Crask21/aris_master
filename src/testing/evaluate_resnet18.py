#!/usr/bin/env python3
# ---------------------------------------------------------------------------- #
#                                    Imports                                   #
# ---------------------------------------------------------------------------- #
import json
import sys
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
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_recall_fscore_support
)
from typing import overload
from resnet_dataloader import ResNetDataloader
import logging
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------- #
#                            Evaluation Functions                              #
# ---------------------------------------------------------------------------- #
def load_model(checkpoint_path, num_classes, device):
    """
    Load ResNet18 model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        num_classes: Number of output classes
        device: torch device (cuda/cpu)
    
    Returns:
        model: Loaded model
        checkpoint: Full checkpoint dict with training history
    """
    # Create model architecture
    model = torchvision.models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    return model, checkpoint


def get_test_loader(config_path, dataloader_instance=None):
    """
    Get test dataloader if available, otherwise use validation dataloader.
    
    Args:
        config_path: Path to config file
        dataloader_instance: Existing ResNetDataloader instance (optional)
    
    Returns:
        test_loader: DataLoader for testing
        split_name: Name of the split being used ('test' or 'val')
        class_names: List of class names
    """
    if dataloader_instance is None:
        dataloader_instance = ResNetDataloader(config_path)
    
    try:
        if dataloader_instance.test_loader is not None:
            return dataloader_instance.test_loader, "test", dataloader_instance.classes
    except AttributeError:
        logger.warning("test_loader not found in dataloader instance, trying val_loader...")
    
    if dataloader_instance.val_loader is not None:
        return dataloader_instance.val_loader, "val", dataloader_instance.classes
    
    else:
        raise ValueError("No test or validation dataloader found in config")

def evaluate_model(model, dataloader, device, class_names):
    """
    Evaluate model on a dataset.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation
        device: torch device
        class_names: List of class names
    
    Returns:
        results: Dictionary containing predictions, labels, filepaths, and metrics
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probs = []
    all_filepaths = []
    
    logger.info(f"Evaluating model on {len(dataloader.dataset)} samples...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs = batch["image"].to(device)
            labels = batch["class"].to(device)
            
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            # Try to get filepaths if available
            if "filepath" in batch:
                all_filepaths.extend(batch["filepath"])
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted'
    )

    # Per-class accuracy
    per_class_accuracy = {}
    for cls_idx, cls_name in enumerate(class_names):
        mask = all_labels == cls_idx
        if mask.sum() > 0:
            per_class_accuracy[cls_name] = float(
                (all_predictions[mask] == cls_idx).sum() / mask.sum()
            )
        else:
            per_class_accuracy[cls_name] = None
    
    results = {
        "predictions": all_predictions,
        "labels": all_labels,
        "probabilities": all_probs,
        "filepaths": all_filepaths,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "class_names": class_names,
        "per_class_accuracy": per_class_accuracy,
    }
    
    logger.info(f"\n[RESULTS]")
    logger.info(f"  Accuracy:  {accuracy*100:.2f}%")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall:    {recall:.4f}")
    logger.info(f"  F1 Score:  {f1:.4f}")
    for cls_name, cls_acc in per_class_accuracy.items():
        if cls_acc is not None:
            logger.info(f"  {cls_name}: {cls_acc*100:.2f}%")
    
    return results


def save_predictions_csv(results, output_path):
    """
    Save predictions to CSV file.
    
    Args:
        results: Results dictionary from evaluate_model
        output_path: Path to save CSV file
    """
    class_names = results["class_names"]
    
    # Create DataFrame
    df_data = {
        "true_label": [class_names[label] for label in results["labels"]],
        "predicted_label": [class_names[pred] for pred in results["predictions"]],
        "correct": results["labels"] == results["predictions"],
        "probabilities": results["probabilities"].tolist()
    }
    
    # Add probability columns for each class
    for i, class_name in enumerate(class_names):
        df_data[f"prob_{class_name}"] = results["probabilities"][:, i]
    
    # Add filepaths if available
    if len(results["filepaths"]) > 0:
        df_data["filepath"] = results["filepaths"]
    
    df = pd.DataFrame(df_data)
    df.to_csv(output_path, index=False)
    logger.info(f"[INFO] Predictions saved to: {output_path}")


def save_confusion_matrix(results, output_path):
    """
    Save confusion matrix as image.
    
    Args:
        results: Results dictionary from evaluate_model
        output_path: Path to save confusion matrix image
    """
    class_names = results["class_names"]
    cm = confusion_matrix(results["labels"], results["predictions"])

    # Normalize by true-label row so color intensity reflects per-class recall distribution.
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_normalized = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    # Build annotations with both percentage and absolute count for each cell.
    annot_labels = np.empty(cm.shape, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot_labels[i, j] = f"{cm_normalized[i, j] * 100:.1f}%\n({cm[i, j]})"

    # Main confusion matrix: color by normalized percentage, annotate with percentage + count.
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_normalized,
        annot=annot_labels,
        fmt='',
        cmap='Blues',
        vmin=0.0,
        vmax=1.0,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Row-normalized percentage'}
    )
    plt.title('Confusion Matrix (Percentage + Count)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"[INFO] Confusion matrix saved to: {output_path}")

    output_path_counts_npy = output_path.replace('.png', '.npy')
    np.save(output_path_counts_npy, cm.astype(float))
    logger.info(f"[INFO] Confusion matrix array saved to: {output_path_counts_npy}")

    # Also save normalized-only confusion matrix for downstream aggregation.
    output_path_norm = output_path.replace('.png', '_normalized.png')
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.1%',
        cmap='Blues',
        vmin=0.0,
        vmax=1.0,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Row-normalized percentage'}
    )
    plt.title('Normalized Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path_norm, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"[INFO] Normalized confusion matrix saved to: {output_path_norm}")

    # Save the normalized matrix as .npy for multi-run aggregation tooling.
    output_path_norm_npy = output_path.replace('.png', '_normalized.npy')
    np.save(output_path_norm_npy, cm_normalized)
    logger.info(f"[INFO] Normalized confusion matrix array saved to: {output_path_norm_npy}")


def save_classification_report(results, output_path):
    """
    Save classification report as text file.
    
    Args:
        results: Results dictionary from evaluate_model
        output_path: Path to save classification report
    """
    class_names = results["class_names"]

    report = classification_report(
        results["labels"],
        results["predictions"],
        target_names=class_names,
        digits=4,
        output_dict=True
    )
    return report


def evaluate_resnet18_from_config(config_path: str, checkpoint_path=None, output_dir=None):
    """
    Main evaluation function that can be called programmatically.
    
    Args:
        config_path: Path to config JSON file
        checkpoint_path: Path to model checkpoint (optional, will use config if None)
        output_dir: Output directory for results (optional, will use config if None)
    
    Returns:
        results: Dictionary containing all evaluation results and metrics
    """
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Set checkpoint path from config if not provided
    if checkpoint_path is None:
        checkpoint_path = config["logging"].get("checkpoint_dir")
        if checkpoint_path is None:
            raise ValueError("No checkpoint path provided and none found in config")
    
    # Set output directory
    if output_dir is None:
        output_dir = config["logging"]["output_dir"] + "/evaluation/"
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[INFO] Using device: {device}")
    
    # Load dataloader
    logger.info("[INFO] Loading dataloader...")
    dataloader_instance = ResNetDataloader(config_path)
    
    # Get test loader (or validation as fallback)
    test_loader, split_name, class_names = get_test_loader(config_path, dataloader_instance)
    num_classes = len(class_names)
    
    # Load model
    model, checkpoint = load_model(checkpoint_path, num_classes, device)
    
    # Print checkpoint info
    if "epoch" in checkpoint:
        logger.info(f"[INFO] Loaded checkpoint from epoch {checkpoint['epoch'] + 1}")
    if "val_acc" in checkpoint and len(checkpoint["val_acc"]) > 0:
        logger.info(f"[INFO] Best validation accuracy during training: {max(checkpoint['val_acc']):.2f}%")
    
    # Evaluate model
    results = evaluate_model(model, test_loader, device, class_names)
    results["split_name"] = split_name
    results["checkpoint_path"] = checkpoint_path
    results["config_path"] = config_path
    
    # Save results
    logger.info(f"[INFO] Saving results to: {output_dir}")
    save_predictions_csv(results, os.path.join(output_dir, f"predictions_{split_name}.csv"))
    save_confusion_matrix(results, os.path.join(output_dir, f"confusion_matrix_{split_name}.png"))
    classification_report_dict = save_classification_report(results, os.path.join(output_dir, f"classification_report_{split_name}.txt"))
    
    # Save summary JSON
    summary = {
        "split": split_name,
        "checkpoint": checkpoint_path,
        "num_samples": len(results["labels"]),
        "num_classes": num_classes,
        "class_names": class_names,
        "accuracy": float(results["accuracy"]),
        "precision": float(results["precision"]),
        "recall": float(results["recall"]),
        "f1_score": float(results["f1"]),
        "per_class_accuracy": results["per_class_accuracy"],
        "classification_report": classification_report_dict
    }
    
    summary_path = os.path.join(output_dir, f"evaluation_summary_{split_name}.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    logger.info(f"[INFO] Evaluation summary saved to: {summary_path}")
    
    logger.warning("\n✓ Evaluation complete!")
    return results



def evaluate_resnet18(dataloader: ResNetDataloader, checkpoint_dir, best_checkpoint="lowest_val_loss", output_dir=None, split_name="val"):
    """
    Main evaluation function that can be called programmatically.
    
    Args:
        dataloader: The dataloader instance to use for evaluation
        checkpoint_path: Path to model checkpoint (optional, will use config if None)
        output_dir: Output directory for results (optional, will use config if None)
        best_checkpoint: Which checkpoint to use if multiple are available ("lowest_val_loss", "highest_val_acc", or a specific epoch number)
    Returns:
        results: Dictionary containing all evaluation results and metrics
    """
    # Load config

    
    # Set checkpoint path from config if not provided
    # Check if checkpoint_path exists, if not try to find checkpoint in config logging directory
    if checkpoint_dir is not None:
        
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
        
        if best_checkpoint is not None:
            if best_checkpoint not in ["lowest_val_loss", "highest_val_acc"] and not isinstance(best_checkpoint, int):
                raise ValueError(f"Invalid best_checkpoint value: {best_checkpoint}. Must be 'lowest_val_loss', 'highest_val_acc', or an integer epoch number.")
            elif best_checkpoint == "lowest_val_loss":
                checkpoint_path = Path(checkpoint_dir) / "resnet18_lowest_val_loss.ckpt"
            elif best_checkpoint == "highest_val_acc":
                checkpoint_path = Path(checkpoint_dir) / "resnet18_best_val_acc.ckpt"
            else:
                checkpoint_path = Path(checkpoint_dir) / f"resnet18_epoch_{best_checkpoint}.ckpt"
        else:
            checkpoint_path = Path(checkpoint_dir) / "resnet18_lowest_val_loss.ckpt"    
    
    if output_dir is None:
        output_dir = os.path.join(checkpoint_dir, "evaluation")
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.debug(f"[INFO] Using device: {device}")
    
    
    # Get test loader (or validation as fallback)
    if split_name == "test":
        test_loader = dataloader.test_loader
    else:
        test_loader = dataloader.val_loader
    
    class_names = dataloader.classes
    num_classes = dataloader.num_classes
    
    # Load model
    model, checkpoint = load_model(checkpoint_path, num_classes, device)
    
    # Print checkpoint info
    if "epoch" in checkpoint:
        logger.info(f"[INFO] Loaded checkpoint from epoch {checkpoint['epoch'] + 1}")
    if "val_acc" in checkpoint and len(checkpoint["val_acc"]) > 0:
        logger.info(f"[INFO] Best validation accuracy during training: {max(checkpoint['val_acc']):.2f}%")
    
    # Evaluate model
    results = evaluate_model(model, test_loader, device, class_names)
    results["split_name"] = split_name
    results["checkpoint_path"] = checkpoint_path
    # results["config_path"] = config_path
    
    # Save results
    logger.debug(f"[INFO] Saving results to: {output_dir}")
    save_predictions_csv(results, os.path.join(output_dir, f"predictions_{split_name}.csv"))
    save_confusion_matrix(results, os.path.join(output_dir, f"confusion_matrix_{split_name}.png"))
    save_classification_report(results, os.path.join(output_dir, f"classification_report_{split_name}.txt"))
    
    # Save summary JSON
    summary = {
        "split": split_name,
        "checkpoint": str(checkpoint_path),
        "num_samples": len(results["labels"]),
        "num_classes": num_classes,
        "class_names": class_names,
        "accuracy": float(results["accuracy"]),
        "precision": float(results["precision"]),
        "recall": float(results["recall"]),
        "f1_score": float(results["f1"]),
        "per_class_accuracy": results["per_class_accuracy"],
    }
    
    summary_path = os.path.join(output_dir, f"evaluation_summary_{split_name}.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    logger.debug(f"[INFO] Evaluation summary saved to: {summary_path}")
    
    logger.info("\n✓ Evaluation complete!")
    return results

def multi_run_evaluation(resnet18_runs_dir, output_dir=None):
    """
    Evaluate multiple ResNet18 runs in a directory and aggregate results.
    
    Args:
        resnet18_runs_dir: Directory containing subdirectories for each run (each with a checkpoint and config)
        output_dir: Directory to save aggregated results (optional, will create 'multi_run_evaluation' in output dir if not provided)
        """
# ---------------------------------------------------------------------------- #
#                                     Main                                     #
# ---------------------------------------------------------------------------- #
def main():
    parser = ArgumentParser(description="Evaluate ResNet18 model on test/validation dataset")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        #required=True,
        help="Path to config JSON file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (optional, will use checkpoint from config if not provided)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for evaluation results (optional, will use config output_dir/evaluation if not provided)"
    )
    parser.add_argument(
        "--resnet18-runs-dir",
        type=str,
        default=None,
        help="Directory containing resnet18 runs."
    )
    
    args = parser.parse_args()
    
    try:
        if args.config or args.checkpoint:
            results = evaluate_resnet18_from_config(
                config_path=args.config,
                checkpoint_path=args.checkpoint,
                output_dir=args.output_dir
            )
        elif args.resnet18_runs_dir:
            multi_run_evaluation(resnet18_runs_dir=args.resnet18_runs_dir, output_dir=args.output_dir)
    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
