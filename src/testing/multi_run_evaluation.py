
from seaborn.objects import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
# Import parser args
from argparse import ArgumentParser
from src.utils.add_note import add_note
import logging
logger = logging.getLogger(__name__)


def multi_run_evaluation(resnet18_runs_dir, output_dir=None):
    """
    Evaluate multiple ResNet18 runs in a directory and aggregate results.
    
    Args:
        resnet18_runs_dir: Directory containing subdirectories for each run (each with a checkpoint and config)
        output_dir: Directory to save aggregated results (optional, will create 'multi_run_evaluation' in output dir if not provided)
    """
    resnet18_runs_dir = Path(resnet18_runs_dir)
    
    if not resnet18_runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {resnet18_runs_dir}")
    
    # Set output directory
    if output_dir is None:
        output_dir = resnet18_runs_dir.parent / "multi_run_evaluation"
    else:
        output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Analyzing runs in: {resnet18_runs_dir}")
    logger.debug(f"Output directory: {output_dir}")
    
    # Find all run directories
    run_dirs = sorted([d for d in resnet18_runs_dir.iterdir() if d.is_dir()])
    
    if not run_dirs:
        raise ValueError(f"No run directories found in: {resnet18_runs_dir}")
    
    logger.info(f"Found {len(run_dirs)} run directories")
    
    # Group runs by split configuration (e.g., "2000-real_0-synthetic")
    splits_data = {}
    
    for run_dir in run_dirs:
        # Parse directory name (e.g., "2000-real_1000-synthetic_run1")
        dir_name = run_dir.name
        
        # Extract split name and run number
        import re
        match = re.match(r'(\d+)-real_(\d+)-synthetic_run(\d+)', dir_name)
        if not match:
            logger.warning(f"Skipping directory with unexpected name format: {dir_name}")
            continue
        
        real_count = int(match.group(1))
        synthetic_count = int(match.group(2))
        run_number = int(match.group(3))
        
        split_name = f"{real_count}-real_{synthetic_count}-synthetic"
        
        # Look for evaluation summary file
        eval_summary_path = run_dir / "evaluation" / "evaluation_summary_val.json"
        
        if not eval_summary_path.exists():
            logger.warning(f"No evaluation summary found for {dir_name}, skipping...")
            continue
        
        # Load evaluation summary
        with open(eval_summary_path, 'r') as f:
            eval_summary = json.load(f)
        
        # Initialize split data if needed
        if split_name not in splits_data:
            splits_data[split_name] = {
                "split_name": split_name,
                "real_image_count": real_count,
                "synthetic_image_count": synthetic_count,
                "runs": []
            }
        
        # Add run data
        run_data = {
            "run_id": f"run{run_number}",
            "checkpoint": eval_summary["checkpoint"],
            "accuracy": eval_summary["accuracy"],
            "precision": eval_summary["precision"],
            "recall": eval_summary["recall"],
            "f1_score": eval_summary["f1_score"]
        }
        
        splits_data[split_name]["runs"].append(run_data)
        logger.info(f"Loaded results for {split_name} run{run_number}: acc={eval_summary['accuracy']:.4f}, f1={eval_summary['f1_score']:.4f}")
    
    # Calculate mean and std for each split
    for split_name, split_data in splits_data.items():
        runs = split_data["runs"]
        
        if len(runs) == 0:
            logger.warning(f"No runs found for split: {split_name}")
            continue
        
        accuracies = [run["accuracy"] for run in runs]
        f1_scores = [run["f1_score"] for run in runs]
        
        split_data["mean_accuracy"] = float(np.mean(accuracies))
        split_data["std_accuracy"] = float(np.std(accuracies, ddof=1))  # Use sample std deviation
        split_data["mean_f1_score"] = float(np.mean(f1_scores))
        split_data["std_f1_score"] = float(np.std(f1_scores, ddof=1))  # Use sample std deviation
        
        logger.info(f"Summary for {split_name}:")
        logger.info(f"  Runs: {len(runs)}")
        logger.info(f"  Mean Accuracy: {split_data['mean_accuracy']:.4f} ± {split_data['std_accuracy']:.4f}")
        logger.info(f"  Mean F1 Score: {split_data['mean_f1_score']:.4f} ± {split_data['std_f1_score']:.4f}")
    
    # Sort splits by real_image_count, then synthetic_image_count
    sorted_splits = sorted(
        splits_data.values(),
        key=lambda x: (x["real_image_count"], x["synthetic_image_count"])
    )
    
    # Create output structure
    multi_run_results = {
        "multi_run_results": {
            "splits": sorted_splits
        }
    }
    
    # Save results
    output_path = output_dir / "multi_run_results.json"
    with open(output_path, 'w') as f:
        json.dump(multi_run_results, f, indent=4)
    
    logger.info(f"Multi-run results saved to: {output_path}")
    
    # Create summary plot
    create_multi_run_plot(sorted_splits, output_dir)
    
    # Create bar chart comparing mean accuracy across splits
    bar_chart(sorted_splits, output_dir)
    
    logger.info("✓ Multi-run evaluation complete!")
    return multi_run_results

def create_multi_run_plot(splits_data, output_dir):
    """
    Create visualization plots for multi-run results.
    
    Args:
        splits_data: List of split data dictionaries
        output_dir: Directory to save plots
    """
    if not splits_data:
        logger.warning("No data to plot")
        return
    
    # Group splits by real image count
    splits_by_real = {}
    for split_data in splits_data:
        real_count = split_data["real_image_count"]
        if real_count not in splits_by_real:
            splits_by_real[real_count] = []
        splits_by_real[real_count].append(split_data)
    
    # Create a plot for each real image count
    for real_count, splits in splits_by_real.items():
        # Sort by synthetic count
        splits = sorted(splits, key=lambda x: x["synthetic_image_count"])
        
        synthetic_counts = [s["synthetic_image_count"] for s in splits]
        mean_accuracies = [s["mean_accuracy"] for s in splits]
        std_accuracies = [s["std_accuracy"] for s in splits]
        mean_f1_scores = [s["mean_f1_score"] for s in splits]
        std_f1_scores = [s["std_f1_score"] for s in splits]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot accuracy with individual runs
        ax1.errorbar(synthetic_counts, mean_accuracies, yerr=std_accuracies, 
                    fmt='o-', markersize=8, capsize=5, linewidth=2, 
                    color='darkblue', markerfacecolor='red', 
                    ecolor='gray', capthick=2, label='Mean ± Std')
        
        # Plot individual runs as scatter points
        for i, split_data in enumerate(splits):
            run_accuracies = [run["accuracy"] for run in split_data["runs"]]
            x_positions = [synthetic_counts[i]] * len(run_accuracies)
            ax1.scatter(x_positions, run_accuracies, alpha=0.4, s=80, 
                       color='skyblue', edgecolors='black', linewidth=0.5, 
                       zorder=2)
        
        ax1.set_xlabel('Number of Synthetic Images', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax1.set_title(f'Accuracy vs Synthetic Images ({real_count} Real Images)', 
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_ylim([0, 1])
        ax1.legend(loc='best')
        
        # Add value labels on the mean points
        for x, y in zip(synthetic_counts, mean_accuracies):
            ax1.annotate(f'{y:.3f}', xy=(x, y), xytext=(0, 10), 
                        textcoords='offset points', ha='center', 
                        fontsize=9, fontweight='bold')
        
        # Plot F1 score with individual runs
        ax2.errorbar(synthetic_counts, mean_f1_scores, yerr=std_f1_scores, 
                    fmt='o-', markersize=8, capsize=5, linewidth=2, 
                    color='darkred', markerfacecolor='red', 
                    ecolor='gray', capthick=2, label='Mean ± Std')
        
        # Plot individual runs as scatter points
        for i, split_data in enumerate(splits):
            run_f1_scores = [run["f1_score"] for run in split_data["runs"]]
            x_positions = [synthetic_counts[i]] * len(run_f1_scores)
            ax2.scatter(x_positions, run_f1_scores, alpha=0.4, s=80, 
                       color='lightcoral', edgecolors='black', linewidth=0.5, 
                       zorder=2)
        
        ax2.set_xlabel('Number of Synthetic Images', fontsize=12, fontweight='bold')
        ax2.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
        ax2.set_title(f'F1 Score vs Synthetic Images ({real_count} Real Images)', 
                     fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_ylim([0, 1])
        ax2.legend(loc='best')
        
        # Add value labels on the mean points
        for x, y in zip(synthetic_counts, mean_f1_scores):
            ax2.annotate(f'{y:.3f}', xy=(x, y), xytext=(0, 10), 
                        textcoords='offset points', ha='center', 
                        fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = output_dir / f"accuracy_f1_vs_synthetic_{real_count}_real.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        add_note(
            notes_path=output_dir / "../notes.md",
            title=f"Accuracy and F1 Score vs Synthetic Images in Connected dot plot",
            content=plot_path
        )
        plt.close()
        
        logger.info(f"Plot saved to: {plot_path}")
    
    # Create combined plot if there are multiple real counts
    if len(splits_by_real) > 1:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(splits_by_real)))
        
        for idx, (real_count, splits) in enumerate(sorted(splits_by_real.items())):
            splits = sorted(splits, key=lambda x: x["synthetic_image_count"])
            
            synthetic_counts = [s["synthetic_image_count"] for s in splits]
            mean_accuracies = [s["mean_accuracy"] for s in splits]
            std_accuracies = [s["std_accuracy"] for s in splits]
            mean_f1_scores = [s["mean_f1_score"] for s in splits]
            std_f1_scores = [s["std_f1_score"] for s in splits]
            
            # Plot accuracy
            ax1.errorbar(synthetic_counts, mean_accuracies, yerr=std_accuracies,
                        fmt='o-', markersize=8, capsize=5, linewidth=2,
                        color=colors[idx], label=f'{real_count} Real Images',
                        capthick=2)
            
            # Plot F1 score
            ax2.errorbar(synthetic_counts, mean_f1_scores, yerr=std_f1_scores,
                        fmt='o-', markersize=8, capsize=5, linewidth=2,
                        color=colors[idx], label=f'{real_count} Real Images',
                        capthick=2)
        
        ax1.set_xlabel('Number of Synthetic Images', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax1.set_title('Accuracy vs Synthetic Images (All Real Counts)', 
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_ylim([0, 1])
        ax1.legend(loc='best')
        
        ax2.set_xlabel('Number of Synthetic Images', fontsize=12, fontweight='bold')
        ax2.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
        ax2.set_title('F1 Score vs Synthetic Images (All Real Counts)', 
                     fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_ylim([0, 1])
        ax2.legend(loc='best')
        
        plt.tight_layout()
        
        combined_plot_path = output_dir / "accuracy_f1_vs_synthetic_combined.png"
        plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
        
        add_note(
            notes_path=output_dir / "../notes.md",
            title=f"Combined Accuracy and F1 Score vs Synthetic Images in Connected dot plot",
            content=combined_plot_path
        )
        plt.close()
        
        logger.info(f"Combined plot saved to: {combined_plot_path}")
    
def bar_chart(splits_data, output_dir):
    """
    Create bar chart comparing mean accuracy across splits.
    
    Args:
        splits_data: List of split data dictionaries
        output_dir: Directory to save plot
    """
    if not splits_data:
        logger.warning("No data to plot")
        return
    
    synthetic_labels = [str(s["synthetic_image_count"]) for s in splits_data]
    real_count = splits_data[0]["real_image_count"]
    mean_accuracies = [s["mean_accuracy"] for s in splits_data]
    std_accuracies = [s["std_accuracy"] for s in splits_data]
    
    plt.figure(figsize=(12, 6))
    plt.bar(synthetic_labels, mean_accuracies, yerr=std_accuracies, capsize=5, color='skyblue', edgecolor='black')
    plt.xlabel('Number of Synthetic Images', fontsize=12, fontweight='bold')
    plt.ylabel('Mean Accuracy', fontsize=12, fontweight='bold')
    plt.title(f'Mean Accuracy Across Splits ({real_count} Real Images)', fontsize=14, fontweight='bold')
    plt.xticks(rotation=0)
    plt.ylim([0, 1])
    plt.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Add value labels on the bars
    for i, (mean_acc, std_acc) in enumerate(zip(mean_accuracies, std_accuracies)):
        plt.annotate(f'{mean_acc:.3f}', xy=(i, mean_acc), xytext=(20, 5), 
                    textcoords='offset points', ha='center', 
                    fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    bar_chart_path = output_dir / "mean_accuracy_bar_chart.png"
    plt.savefig(bar_chart_path, dpi=300)
    add_note(
            notes_path=output_dir / "../notes.md",
            title=f"Mean Accuracy Bar Chart",
            content=bar_chart_path
        )
    plt.close()
    
    logger.info(f"Bar chart saved to: {bar_chart_path}")
if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluate multiple ResNet18 runs and aggregate results")
    parser.add_argument(
        "--runs_dir",
        type=str,
        default=None,
        help="Directory containing subdirectories for each ResNet18 run (each with a checkpoint and config)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save aggregated results and plots (optional, will create 'multi_run_evaluation' in runs dir if not provided)"
    )
    #args = ["--runs_dir", "/home/ap/cloud/Master/aris_master/testing/02-18_normal-wood_impregnated-wood_splits/resnet18_runs"]
    args = parser.parse_args()
    multi_run_evaluation(args.runs_dir, args.output_dir)