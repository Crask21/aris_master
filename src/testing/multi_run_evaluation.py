
# from seaborn.objects import Path
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
# Import parser args
from argparse import ArgumentParser
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    
from src.utils.add_note import add_note
import logging
logger = logging.getLogger(__name__)


def _extract_class_names_from_report(classification_report):
    """Extract class labels from sklearn classification report dict."""
    if not isinstance(classification_report, dict):
        return []

    excluded = {"accuracy", "macro avg", "weighted avg", "micro avg", "samples avg"}
    class_names = []
    for key, value in classification_report.items():
        if key in excluded:
            continue
        if isinstance(value, dict) and "support" in value:
            class_names.append(str(key))
    return class_names


def _extract_class_supports(classification_report, class_names):
    """Return per-class supports in class_names order."""
    if not isinstance(classification_report, dict) or not class_names:
        return None

    supports = []
    for class_name in class_names:
        class_metrics = classification_report.get(class_name)
        if not isinstance(class_metrics, dict) or "support" not in class_metrics:
            return None
        supports.append(float(class_metrics["support"]))
    return np.asarray(supports, dtype=float)


def _load_confusion_matrices(run_dir, evaluation_split, eval_summary):
    """
    Load confusion matrices for one run.

    Returns:
        cm_counts: np.ndarray or None
        cm_normalized: np.ndarray or None
        class_names: list[str]
    """
    eval_dir = run_dir / "evaluation"
    cm_counts_path = eval_dir / f"confusion_matrix_{evaluation_split}.npy"
    cm_normalized_path = eval_dir / f"confusion_matrix_{evaluation_split}_normalized.npy"

    cm_counts = None
    cm_normalized = None

    if cm_counts_path.exists():
        cm_counts = np.load(cm_counts_path).astype(float)

    if cm_normalized_path.exists():
        cm_normalized = np.load(cm_normalized_path).astype(float)

    class_names = _extract_class_names_from_report(eval_summary.get("classification_report", {}))

    # Recover missing normalized matrix from counts if needed.
    if cm_normalized is None and cm_counts is not None:
        row_sums = cm_counts.sum(axis=1, keepdims=True)
        cm_normalized = np.divide(
            cm_counts,
            row_sums,
            out=np.zeros_like(cm_counts, dtype=float),
            where=row_sums != 0
        )

    # Recover missing counts matrix from normalized matrix and per-class supports.
    if cm_counts is None and cm_normalized is not None:
        supports = _extract_class_supports(eval_summary.get("classification_report", {}), class_names)
        if supports is not None and cm_normalized.shape[0] == len(supports):
            cm_counts = cm_normalized * supports[:, np.newaxis]

    return cm_counts, cm_normalized, class_names


def _plot_mean_confusion_matrix(mean_cm_counts, mean_cm_normalized, class_names, output_path, title):
    """
    Plot confusion matrix with normalized coloring and percentage+count annotations.
    """
    if mean_cm_counts is None or mean_cm_normalized is None:
        return

    if not class_names or len(class_names) != mean_cm_normalized.shape[0]:
        class_names = [str(i) for i in range(mean_cm_normalized.shape[0])]

    annot = np.empty(mean_cm_normalized.shape, dtype=object)
    for i in range(mean_cm_normalized.shape[0]):
        for j in range(mean_cm_normalized.shape[1]):
            annot[i, j] = f"{mean_cm_normalized[i, j] * 100:.1f}%\n({mean_cm_counts[i, j]:.1f})"

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        mean_cm_normalized,
        annot=annot,
        fmt="",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Row-normalized percentage"}
    )
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def multi_run_evaluation(resnet18_runs_dir, output_dir=None, synthetic_real_factor=False, only_show_mean=False, splits=None):
    """
    Evaluate multiple ResNet18 runs in a directory and aggregate results.
    
    Args:
        resnet18_runs_dir: Directory containing subdirectories for each run (each with a checkpoint and config)
        output_dir: Directory to save aggregated results (optional, will create 'multi_run_evaluation' in output dir if not provided)
        synthetic_real_factor: If True, create plots with synthetic/real ratio instead of synthetic/total
        only_show_mean: If True, only show mean ± std without individual run scatter points
        splits: List of evaluation splits to process (e.g., ["val", "test"]). Defaults to ["val"].
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
    if splits is None:
        splits = ["val"]

    split_summaries = {
        "val": "evaluation_summary_val.json",
        "test": "evaluation_summary_test.json",
    }

    aggregated_results = {}

    for evaluation_split in splits:
        summary_filename = split_summaries.get(evaluation_split, f"evaluation_summary_{evaluation_split}.json")
        splits_data = {}

        for run_dir in run_dirs:
            dir_name = run_dir.name
            import re
            match = re.match(r'(\d+)-real_(\d+)-synthetic_run(\d+)', dir_name)
            if not match:
                logger.warning(f"Skipping directory with unexpected name format: {dir_name}")
                continue

            real_count = int(match.group(1))
            synthetic_count = int(match.group(2))
            run_number = int(match.group(3))

            split_name = f"{real_count}-real_{synthetic_count}-synthetic"

            eval_summary_path = run_dir / "evaluation" / summary_filename
            if not eval_summary_path.exists():
                logger.warning(f"No evaluation summary found for {dir_name} ({evaluation_split}), skipping...")
                continue

            with open(eval_summary_path, 'r') as f:
                eval_summary = json.load(f)

            if split_name not in splits_data:
                splits_data[split_name] = {
                    "split_name": split_name,
                    "real_image_count": real_count,
                    "synthetic_image_count": synthetic_count,
                    "runs": []
                }

            run_data = {
                "run_id": f"run{run_number}",
                "checkpoint": eval_summary["checkpoint"],
                "accuracy": eval_summary["accuracy"],
                "precision": eval_summary["precision"],
                "recall": eval_summary["recall"],
                "f1_score": eval_summary["f1_score"]
            }

            cm_counts, cm_normalized, class_names = _load_confusion_matrices(run_dir, evaluation_split, eval_summary)
            run_data["cm_counts"] = cm_counts.tolist() if cm_counts is not None else None
            run_data["cm_normalized"] = cm_normalized.tolist() if cm_normalized is not None else None
            run_data["class_names"] = class_names

            splits_data[split_name]["runs"].append(run_data)
            logger.info(f"Loaded results for {split_name} run{run_number} ({evaluation_split}): acc={eval_summary['accuracy']:.4f}, f1={eval_summary['f1_score']:.4f}")

        # Calculate mean and std for each split
        for split_name, split_data in splits_data.items():
            runs = split_data["runs"]
            if len(runs) == 0:
                logger.warning(f"No runs found for split: {split_name} ({evaluation_split})")
                continue
            accuracies = [run["accuracy"] for run in runs]
            f1_scores = [run["f1_score"] for run in runs]
            split_data["mean_accuracy"] = float(np.mean(accuracies))
            split_data["std_accuracy"] = float(np.std(accuracies, ddof=1))
            split_data["mean_f1_score"] = float(np.mean(f1_scores))
            split_data["std_f1_score"] = float(np.std(f1_scores, ddof=1))

            # Aggregate confusion matrices (counts and percentages) across runs.
            cm_counts_list = [np.array(run["cm_counts"], dtype=float) for run in runs if run.get("cm_counts") is not None]
            cm_norm_list = [np.array(run["cm_normalized"], dtype=float) for run in runs if run.get("cm_normalized") is not None]

            # Keep only matrices with matching shapes.
            if cm_counts_list:
                target_shape = cm_counts_list[0].shape
                cm_counts_list = [cm for cm in cm_counts_list if cm.shape == target_shape]
            if cm_norm_list:
                target_shape = cm_norm_list[0].shape
                cm_norm_list = [cm for cm in cm_norm_list if cm.shape == target_shape]

            if cm_counts_list and not cm_norm_list:
                cm_norm_list = []
                for cm_counts in cm_counts_list:
                    row_sums = cm_counts.sum(axis=1, keepdims=True)
                    cm_norm = np.divide(
                        cm_counts,
                        row_sums,
                        out=np.zeros_like(cm_counts, dtype=float),
                        where=row_sums != 0
                    )
                    cm_norm_list.append(cm_norm)

            if cm_counts_list and cm_norm_list and cm_counts_list[0].shape == cm_norm_list[0].shape:
                mean_cm_counts = np.mean(cm_counts_list, axis=0)
                mean_cm_normalized = np.mean(cm_norm_list, axis=0)

                split_data["mean_confusion_matrix_counts"] = mean_cm_counts.tolist()
                split_data["mean_confusion_matrix_percentages"] = mean_cm_normalized.tolist()

                class_names = []
                for run in runs:
                    candidate = run.get("class_names")
                    if candidate and len(candidate) == mean_cm_normalized.shape[0]:
                        class_names = candidate
                        break
                split_data["class_names"] = class_names

                cm_counts_save_path = output_dir / f"mean_confusion_matrix_counts_{split_name}_{evaluation_split}.npy"
                cm_percent_save_path = output_dir / f"mean_confusion_matrix_percentages_{split_name}_{evaluation_split}.npy"
                np.save(cm_counts_save_path, mean_cm_counts)
                np.save(cm_percent_save_path, mean_cm_normalized)

                cm_plot_path = output_dir / f"mean_confusion_matrix_{split_name}_{evaluation_split}.png"
                _plot_mean_confusion_matrix(
                    mean_cm_counts=mean_cm_counts,
                    mean_cm_normalized=mean_cm_normalized,
                    class_names=class_names,
                    output_path=cm_plot_path,
                    title=f"Mean Confusion Matrix ({evaluation_split})\n{split_name}"
                )
                add_note(
                    notes_path=output_dir / "../notes.md",
                    title=f"Mean confusion matrix ({split_name}, {evaluation_split})",
                    content=cm_plot_path
                )
                logger.info(f"Mean confusion matrix saved to: {cm_plot_path}")
            else:
                logger.warning(
                    f"No compatible confusion matrix data for {split_name} ({evaluation_split}). "
                    "Expected confusion_matrix_<split>.npy and/or confusion_matrix_<split>_normalized.npy per run."
                )

            logger.info(f"Summary for {split_name} ({evaluation_split}):")
            logger.info(f"  Runs: {len(runs)}")
            logger.info(f"  Mean Accuracy: {split_data['mean_accuracy']:.4f} ± {split_data['std_accuracy']:.4f}")
            logger.info(f"  Mean F1 Score: {split_data['mean_f1_score']:.4f} ± {split_data['std_f1_score']:.4f}")

        sorted_splits = sorted(
            splits_data.values(),
            key=lambda x: (x["real_image_count"], x["synthetic_image_count"])
        )
        aggregated_results[evaluation_split] = sorted_splits

        if sorted_splits:
            create_multi_run_plot(sorted_splits, output_dir, synthetic_real_factor, only_show_mean, evaluation_split=evaluation_split)
            bar_chart(sorted_splits, output_dir, evaluation_split=evaluation_split)

    # Output structure for backward compatibility
    multi_run_results = {
        "multi_run_results": {
            "splits": aggregated_results.get("val", []),
            "test_splits": aggregated_results.get("test", [])
        }
    }
    output_path = output_dir / "multi_run_results.json"
    with open(output_path, 'w') as f:
        json.dump(multi_run_results, f, indent=4)
    logger.info(f"Multi-run results saved to: {output_path}")
    logger.info("✓ Multi-run evaluation complete!")
    return multi_run_results

def create_multi_run_plot(splits_data, output_dir, synthetic_real_factor=False, only_show_mean=False, evaluation_split=None):
    """
    Create visualization plots for multi-run results.
    
    Args:
        splits_data: List of split data dictionaries
        output_dir: Directory to save plots
        synthetic_real_factor: If True, create plots with synthetic/real ratio instead of synthetic/total
        only_show_mean: If True, only show mean ± std without individual run scatter points
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
    
    # Create plots for each real image count
    for real_count, splits in splits_by_real.items():
        # Sort by synthetic count
        splits = sorted(splits, key=lambda x: x["synthetic_image_count"])
        
        synthetic_counts = [s["synthetic_image_count"] for s in splits]
        mean_accuracies = [s["mean_accuracy"] for s in splits]
        std_accuracies = [s["std_accuracy"] for s in splits]
        mean_f1_scores = [s["mean_f1_score"] for s in splits]
        std_f1_scores = [s["std_f1_score"] for s in splits]
        
        # Calculate factor based on flag
        if synthetic_real_factor:
            # synthetic/real
            factors = [s / real_count if real_count > 0 else 0 for s in synthetic_counts]
            factor_label = 'Synthetic/Real Ratio'
        else:
            # synthetic/total (default)
            factors = [s / (s + real_count) if (s + real_count) > 0 else 0 for s in synthetic_counts]
            # Round factors to 3 decimal places for better x-tick labels
            factors = [round(f, 3) for f in factors]
            factor_label = 'Synthetic/Total Ratio'
        
        # === ACCURACY PLOT ===
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot individual runs as scatter points (behind error bars)
        if not only_show_mean:
            for i, split_data in enumerate(splits):
                run_accuracies = [run["accuracy"] for run in split_data["runs"]]
                x_positions = [synthetic_counts[i]] * len(run_accuracies)
                ax.scatter(x_positions, run_accuracies, alpha=0.4, s=80, 
                           color='skyblue', edgecolors='black', linewidth=0.5, 
                           zorder=1)
        
        # Plot accuracy with error bars (on top)
        ax.errorbar(synthetic_counts, mean_accuracies, yerr=std_accuracies, 
                    fmt='o-', markersize=8, capsize=5, linewidth=2, 
                    color='darkblue', markerfacecolor='blue', 
                    ecolor='gray', capthick=2, label='Mean ± Std', zorder=3)
        
        ax.set_xlabel('Number of Synthetic Images', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title(f'Accuracy vs Synthetic Images ({real_count} Real Images)', 
                     fontsize=14, fontweight='bold')
        ax.set_xticks(synthetic_counts)
        ax.grid(True, alpha=0.3, linestyle='--', axis='both')
        ax.set_axisbelow(True)
        ax.legend(loc='best')
        
        # Compute y-limits with padding
        if std_accuracies:
            y_min = min(m - s for m, s in zip(mean_accuracies, std_accuracies))
            y_max = max(m + s for m, s in zip(mean_accuracies, std_accuracies))
        else:
            y_min = min(mean_accuracies)
            y_max = max(mean_accuracies)
        pad = max(0.02, (y_max - y_min) * 0.1)
        ax.set_ylim(y_min - pad, y_max + pad)
        
        # Add value labels on the mean points
        for x, y in zip(synthetic_counts, mean_accuracies):
            ax.annotate(f'{y:.3f}', xy=(x, y), xytext=(0, 10), 
                        textcoords='offset points', ha='center', 
                        fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        suffix = f"_{evaluation_split}" if evaluation_split else ""
        accuracy_plot_path = output_dir / f"accuracy_vs_synthetic_{real_count}_real{suffix}.png"
        plt.savefig(accuracy_plot_path, dpi=300, bbox_inches='tight')
        add_note(
            notes_path=output_dir / "../notes.md",
            title=f"Accuracy vs Synthetic Images ({real_count} real images)",
            content=accuracy_plot_path
        )
        plt.close()
        logger.info(f"Accuracy plot saved to: {accuracy_plot_path}")
        
        # === F1 SCORE PLOT ===
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot individual runs as scatter points (behind error bars)
        if not only_show_mean:
            for i, split_data in enumerate(splits):
                run_f1_scores = [run["f1_score"] for run in split_data["runs"]]
                x_positions = [synthetic_counts[i]] * len(run_f1_scores)
                ax.scatter(x_positions, run_f1_scores, alpha=0.4, s=80, 
                           color='lightcoral', edgecolors='black', linewidth=0.5, 
                           zorder=1)
        
        # Plot F1 score with error bars (on top)
        ax.errorbar(synthetic_counts, mean_f1_scores, yerr=std_f1_scores, 
                    fmt='o-', markersize=8, capsize=5, linewidth=2, 
                    color='darkred', markerfacecolor='red', 
                    ecolor='gray', capthick=2, label='Mean ± Std', zorder=3)
        
        ax.set_xlabel('Number of Synthetic Images', fontsize=12, fontweight='bold')
        ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
        ax.set_title(f'F1 Score vs Synthetic Images ({real_count} Real Images)', 
                     fontsize=14, fontweight='bold')
        ax.set_xticks(synthetic_counts)
        ax.grid(True, alpha=0.3, linestyle='--', axis='both')
        ax.set_axisbelow(True)
        ax.legend(loc='best')
        
        # Compute y-limits with padding
        if std_f1_scores:
            y_min = min(m - s for m, s in zip(mean_f1_scores, std_f1_scores))
            y_max = max(m + s for m, s in zip(mean_f1_scores, std_f1_scores))
        else:
            y_min = min(mean_f1_scores)
            y_max = max(mean_f1_scores)
        pad = max(0.02, (y_max - y_min) * 0.1)
        ax.set_ylim(y_min - pad, y_max + pad)
        
        # Add value labels on the mean points
        for x, y in zip(synthetic_counts, mean_f1_scores):
            ax.annotate(f'{y:.3f}', xy=(x, y), xytext=(0, 10), 
                        textcoords='offset points', ha='center', 
                        fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        f1_plot_path = output_dir / f"f1_vs_synthetic_{real_count}_real{suffix}.png"
        plt.savefig(f1_plot_path, dpi=300, bbox_inches='tight')
        add_note(
            notes_path=output_dir / "../notes.md",
            title=f"F1 Score vs Synthetic Images ({real_count} real images)",
            content=f1_plot_path
        )
        plt.close()
        logger.info(f"F1 plot saved to: {f1_plot_path}")
        
        # === FACTOR-BASED ACCURACY PLOT ===
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot individual runs as scatter points (behind error bars)
        if not only_show_mean:
            for i, split_data in enumerate(splits):
                run_accuracies = [run["accuracy"] for run in split_data["runs"]]
                x_positions = [factors[i]] * len(run_accuracies)
                ax.scatter(x_positions, run_accuracies, alpha=0.4, s=80, 
                           color='skyblue', edgecolors='black', linewidth=0.5, 
                           zorder=1)
        
        # Plot factor-based accuracy with error bars (on top)
        ax.errorbar(factors, mean_accuracies, yerr=std_accuracies, 
                    fmt='o-', markersize=8, capsize=5, linewidth=2, 
                    color='darkblue', markerfacecolor='blue', 
                    ecolor='gray', capthick=2, label='Mean ± Std', zorder=3)
        
        ax.set_xlabel(factor_label, fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title(f'Accuracy vs {factor_label} ({real_count} Real Images)', 
                     fontsize=14, fontweight='bold')
        ax.set_xticks(factors)
        ax.grid(True, alpha=0.3, linestyle='--', axis='both')
        ax.set_axisbelow(True)
        ax.legend(loc='best')
        
        # Compute y-limits with padding
        if std_accuracies:
            y_min = min(m - s for m, s in zip(mean_accuracies, std_accuracies))
            y_max = max(m + s for m, s in zip(mean_accuracies, std_accuracies))
        else:
            y_min = min(mean_accuracies)
            y_max = max(mean_accuracies)
        pad = max(0.02, (y_max - y_min) * 0.1)
        ax.set_ylim(y_min - pad, y_max + pad)
        
        # Add value labels on the mean points
        for x, y in zip(factors, mean_accuracies):
            ax.annotate(f'{y:.3f}', xy=(x, y), xytext=(0, 10), 
                        textcoords='offset points', ha='center', 
                        fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        factor_plot_path = output_dir / f"accuracy_vs_factor_{real_count}_real{suffix}.png"
        plt.savefig(factor_plot_path, dpi=300, bbox_inches='tight')
        add_note(
            notes_path=output_dir / "../notes.md",
            title=f"Accuracy vs {factor_label} ({real_count} real images)",
            content=factor_plot_path
        )
        plt.close()
        logger.info(f"Factor-based accuracy plot saved to: {factor_plot_path}")
        
        # === FACTOR-BASED F1 SCORE PLOT ===
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot individual runs as scatter points (behind error bars)
        if not only_show_mean:
            for i, split_data in enumerate(splits):
                run_f1_scores = [run["f1_score"] for run in split_data["runs"]]
                x_positions = [factors[i]] * len(run_f1_scores)
                ax.scatter(x_positions, run_f1_scores, alpha=0.4, s=80, 
                           color='lightcoral', edgecolors='black', linewidth=0.5, 
                           zorder=1)
        
        # Plot factor-based F1 score with error bars (on top)
        ax.errorbar(factors, mean_f1_scores, yerr=std_f1_scores, 
                    fmt='o-', markersize=8, capsize=5, linewidth=2, 
                    color='darkred', markerfacecolor='red', 
                    ecolor='gray', capthick=2, label='Mean ± Std', zorder=3)
        
        ax.set_xlabel(factor_label, fontsize=12, fontweight='bold')
        ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
        ax.set_title(f'F1 Score vs {factor_label} ({real_count} Real Images)', 
                     fontsize=14, fontweight='bold')
        ax.set_xticks(factors)
        ax.grid(True, alpha=0.3, linestyle='--', axis='both')
        ax.set_axisbelow(True)
        ax.legend(loc='best')
        
        # Compute y-limits with padding
        if std_f1_scores:
            y_min = min(m - s for m, s in zip(mean_f1_scores, std_f1_scores))
            y_max = max(m + s for m, s in zip(mean_f1_scores, std_f1_scores))
        else:
            y_min = min(mean_f1_scores)
            y_max = max(mean_f1_scores)
        pad = max(0.02, (y_max - y_min) * 0.1)
        ax.set_ylim(y_min - pad, y_max + pad)
        
        # Add value labels on the mean points
        for x, y in zip(factors, mean_f1_scores):
            ax.annotate(f'{y:.3f}', xy=(x, y), xytext=(0, 10), 
                        textcoords='offset points', ha='center', 
                        fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        factor_f1_plot_path = output_dir / f"f1_vs_factor_{real_count}_real{suffix}.png"
        plt.savefig(factor_f1_plot_path, dpi=300, bbox_inches='tight')
        add_note(
            notes_path=output_dir / "../notes.md",
            title=f"F1 Score vs {factor_label} ({real_count} real images)",
            content=factor_f1_plot_path
        )
        plt.close()
        logger.info(f"Factor-based F1 plot saved to: {factor_f1_plot_path}")
    
    # Create combined plot if there are multiple real counts
    if len(splits_by_real) > 1:
        colors = plt.cm.tab10(np.linspace(0, 1, len(splits_by_real)))
        
        # Collect all values for y-limit calculation and unique x values
        all_mean_accuracies = []
        all_std_accuracies = []
        all_mean_f1_scores = []
        all_std_f1_scores = []
        all_synthetic_counts = set()
        all_factors = set()
        
        # Store data per real_count for plotting
        plot_data = []
        
        for idx, (real_count, splits) in enumerate(sorted(splits_by_real.items())):
            splits = sorted(splits, key=lambda x: x["synthetic_image_count"])
            
            synthetic_counts = [s["synthetic_image_count"] for s in splits]
            mean_accuracies = [s["mean_accuracy"] for s in splits]
            std_accuracies = [s["std_accuracy"] for s in splits]
            mean_f1_scores = [s["mean_f1_score"] for s in splits]
            std_f1_scores = [s["std_f1_score"] for s in splits]
            
            # Calculate factor based on flag
            if synthetic_real_factor:
                factors = [s / real_count if real_count > 0 else 0 for s in synthetic_counts]
            else:
                # Convert to percentage for synthetic/total
                factors = [100 * s / (s + real_count) if (s + real_count) > 0 else 0 for s in synthetic_counts]
            
            # Collect for y-limits
            all_mean_accuracies.extend(mean_accuracies)
            all_std_accuracies.extend(std_accuracies)
            all_mean_f1_scores.extend(mean_f1_scores)
            all_std_f1_scores.extend(std_f1_scores)
            all_synthetic_counts.update(synthetic_counts)
            all_factors.update(factors)
            
            plot_data.append({
                'real_count': real_count,
                'synthetic_counts': synthetic_counts,
                'mean_accuracies': mean_accuracies,
                'std_accuracies': std_accuracies,
                'mean_f1_scores': mean_f1_scores,
                'std_f1_scores': std_f1_scores,
                'factors': factors,
                'color': colors[idx]
            })
        
        # Compute combined y-limits from all data
        if all_std_accuracies:
            acc_y_min = min(m - s for m, s in zip(all_mean_accuracies, all_std_accuracies))
            acc_y_max = max(m + s for m, s in zip(all_mean_accuracies, all_std_accuracies))
        else:
            acc_y_min = min(all_mean_accuracies)
            acc_y_max = max(all_mean_accuracies)
        acc_pad = max(0.02, (acc_y_max - acc_y_min) * 0.1)
        
        if all_std_f1_scores:
            f1_y_min = min(m - s for m, s in zip(all_mean_f1_scores, all_std_f1_scores))
            f1_y_max = max(m + s for m, s in zip(all_mean_f1_scores, all_std_f1_scores))
        else:
            f1_y_min = min(all_mean_f1_scores)
            f1_y_max = max(all_mean_f1_scores)
        f1_pad = max(0.02, (f1_y_max - f1_y_min) * 0.1)
        
        # Determine factor label
        if synthetic_real_factor:
            factor_label = 'Synthetic/Real Ratio'
        else:
            factor_label = 'Synthetic/Total Ratio (%)'
        
        # Sort x-axis values
        sorted_synthetic_counts = sorted(all_synthetic_counts)
        sorted_factors = sorted(all_factors)
        
        # === COMBINED ACCURACY PLOT (by synthetic count) ===
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for data in plot_data:
            ax.errorbar(data['synthetic_counts'], data['mean_accuracies'], 
                       yerr=data['std_accuracies'],
                       fmt='o-', markersize=8, capsize=5, linewidth=2,
                       color=data['color'], markerfacecolor='blue', label=f'{data["real_count"]} Real Images',
                       capthick=2)
        
        ax.set_xlabel('Number of Synthetic Images', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Accuracy vs Synthetic Images (All Real Counts)', 
                     fontsize=14, fontweight='bold')
        ax.set_xticks(sorted_synthetic_counts)
        ax.grid(True, alpha=0.3, linestyle='--', axis='both')
        ax.set_axisbelow(True)
        ax.legend(loc='best')
        ax.set_ylim(acc_y_min - acc_pad, acc_y_max + acc_pad)
        
        plt.tight_layout()
        combined_acc_plot_path = output_dir / f"accuracy_vs_synthetic_combined{suffix}.png"
        plt.savefig(combined_acc_plot_path, dpi=300, bbox_inches='tight')
        add_note(
            notes_path=output_dir / "../notes.md",
            title=f"Combined Accuracy vs Synthetic Images",
            content=combined_acc_plot_path
        )
        plt.close()
        logger.info(f"Combined accuracy plot saved to: {combined_acc_plot_path}")
        
        # === COMBINED F1 PLOT (by synthetic count) ===
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for data in plot_data:
            ax.errorbar(data['synthetic_counts'], data['mean_f1_scores'], 
                       yerr=data['std_f1_scores'],
                       fmt='o-', markersize=8, capsize=5, linewidth=2,
                       color=data['color'], label=f'{data["real_count"]} Real Images',
                       capthick=2)
        
        ax.set_xlabel('Number of Synthetic Images', fontsize=12, fontweight='bold')
        ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
        ax.set_title('F1 Score vs Synthetic Images (All Real Counts)', 
                     fontsize=14, fontweight='bold')
        ax.set_xticks(sorted_synthetic_counts)
        ax.grid(True, alpha=0.3, linestyle='--', axis='both')
        ax.set_axisbelow(True)
        ax.legend(loc='best')
        ax.set_ylim(f1_y_min - f1_pad, f1_y_max + f1_pad)
        
        plt.tight_layout()
        combined_f1_plot_path = output_dir / f"f1_vs_synthetic_combined{suffix}.png"
        plt.savefig(combined_f1_plot_path, dpi=300, bbox_inches='tight')
        add_note(
            notes_path=output_dir / "../notes.md",
            title=f"Combined F1 Score vs Synthetic Images",
            content=combined_f1_plot_path
        )
        plt.close()
        logger.info(f"Combined F1 plot saved to: {combined_f1_plot_path}")
        
        # === COMBINED ACCURACY PLOT (by factor) ===
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for data in plot_data:
            ax.errorbar(data['factors'], data['mean_accuracies'], 
                       yerr=data['std_accuracies'],
                       fmt='o-', markersize=8, capsize=5, linewidth=2,
                       color=data['color'], markerfacecolor='blue', label=f'{data["real_count"]} Real Images',
                       capthick=2)
        
        ax.set_xlabel(factor_label, fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title(f'Accuracy vs {factor_label} (All Real Counts)', 
                     fontsize=14, fontweight='bold')
        ax.set_xticks(sorted_factors)
        ax.grid(True, alpha=0.3, linestyle='--', axis='both')
        ax.set_axisbelow(True)
        ax.legend(loc='best')
        ax.set_ylim(acc_y_min - acc_pad, acc_y_max + acc_pad)
        
        plt.tight_layout()
        combined_factor_acc_path = output_dir / f"accuracy_vs_factor_combined{suffix}.png"
        plt.savefig(combined_factor_acc_path, dpi=300, bbox_inches='tight')
        add_note(
            notes_path=output_dir / "../notes.md",
            title=f"Combined Accuracy vs {factor_label}",
            content=combined_factor_acc_path
        )
        plt.close()
        logger.info(f"Combined factor-based accuracy plot saved to: {combined_factor_acc_path}")
        
        # === COMBINED F1 PLOT (by factor) ===
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for data in plot_data:
            ax.errorbar(data['factors'], data['mean_f1_scores'], 
                       yerr=data['std_f1_scores'],
                       fmt='o-', markersize=8, capsize=5, linewidth=2,
                       color=data['color'], label=f'{data["real_count"]} Real Images',
                       capthick=2)
        
        ax.set_xlabel(factor_label, fontsize=12, fontweight='bold')
        ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
        ax.set_title(f'F1 Score vs {factor_label} (All Real Counts)', 
                     fontsize=14, fontweight='bold')
        ax.set_xticks(sorted_factors)
        ax.grid(True, alpha=0.3, linestyle='--', axis='both')
        ax.set_axisbelow(True)
        ax.legend(loc='best')
        ax.set_ylim(f1_y_min - f1_pad, f1_y_max + f1_pad)
        
        plt.tight_layout()
        combined_factor_f1_path = output_dir / f"f1_vs_factor_combined{suffix}.png"
        plt.savefig(combined_factor_f1_path, dpi=300, bbox_inches='tight')
        add_note(
            notes_path=output_dir / "../notes.md",
            title=f"Combined F1 Score vs {factor_label}",
            content=combined_factor_f1_path
        )
        plt.close()
        logger.info(f"Combined factor-based F1 plot saved to: {combined_factor_f1_path}")
    
def bar_chart(splits_data, output_dir, evaluation_split=None):
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
    
    suffix = f"_{evaluation_split}" if evaluation_split else ""
    bar_chart_path = output_dir / f"mean_accuracy_bar_chart{suffix}.png"
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
    parser.add_argument(
        "--synthetic-real-factor",
        action="store_true",
        help="Use synthetic/real ratio instead of synthetic/total for factor plots"
    )
    parser.add_argument(
        "--only-show-mean",
        action="store_true",
        help="Only show mean ± std without individual run scatter points"
    )
    parser.add_argument(
        "--splits",
        nargs='+',
        default=["val"],
        help="Evaluation splits to process (e.g., val test). Default: val only."
    )
    args = parser.parse_args()
    multi_run_evaluation(args.runs_dir, args.output_dir, args.synthetic_real_factor, args.only_show_mean, splits=args.splits)