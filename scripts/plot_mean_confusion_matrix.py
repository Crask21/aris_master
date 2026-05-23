#!/usr/bin/env python3
"""
Plot a mean confusion matrix from prediction CSV files in a ResNet18 runs directory.

Defaults are intentionally collected near the top so the script can be used by
editing this file or by passing CLI arguments.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from collections import OrderedDict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


DEFAULT_CSV_NAME = "predictions_test.csv"
DEFAULT_TRUE_COLUMN = "true_label"
DEFAULT_PRED_COLUMN = "predicted_label"
DEFAULT_CLASS_LABELS: list[str] | None = None
DEFAULT_DISPLAY_LABELS = {
    "impregnated_wood": "Impregnated Wood",
    "normal_wood": "Normal Wood",
    "soft_plastic": "Soft Plastic",
    "hard_plastic": "Hard Plastic",
}

DEFAULT_TITLE = "Mean Confusion Matrix"
DEFAULT_X_LABEL = "Predicted Label"
DEFAULT_Y_LABEL = "True Label"
DEFAULT_COLORBAR_LABEL = ""

DEFAULT_OUTPUT_NAME = "mean_confusion_matrix_test.png"
DEFAULT_OUTPUT_FORMATS: list[str] | None = None
DEFAULT_FONT_PATH = Path("figures/font/cmunrm.ttf")
DEFAULT_CMAP = "Blues"
DEFAULT_FIGSIZE = (10.0, 8.0)
DEFAULT_DPI = 300

DEFAULT_TITLE_FONT_SIZE = 20
DEFAULT_AXIS_LABEL_FONT_SIZE = 20
DEFAULT_TICK_FONT_SIZE = 17
DEFAULT_ANNOT_FONT_SIZE = 17
DEFAULT_COLORBAR_FONT_SIZE = 17
DEFAULT_PERCENT_DECIMALS = 2
DEFAULT_COUNT_DECIMALS = 1


logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_path(path: str | Path, base: Path | None = None) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (base or _repo_root()) / path


def _configure_font(font_path: Path) -> str:
    if not font_path.exists():
        raise FileNotFoundError(f"Font file not found: {font_path}")

    font_manager.fontManager.addfont(str(font_path))
    font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update(
        {
            "font.family": font_name,
            "font.serif": [font_name],
            "axes.unicode_minus": False,
        }
    )
    return font_name


def _split_name_from_csv(csv_path: Path) -> str | None:
    name = csv_path.name
    prefix = "predictions_"
    suffix = ".csv"
    if name.startswith(prefix) and name.endswith(suffix):
        return name[len(prefix) : -len(suffix)]
    return None


def _load_summary_class_names(csv_path: Path) -> list[str]:
    split_name = _split_name_from_csv(csv_path)
    if not split_name:
        return []

    summary_path = csv_path.parent / f"evaluation_summary_{split_name}.json"
    if not summary_path.exists():
        return []

    try:
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
    except json.JSONDecodeError:
        logger.warning("Could not parse class names from %s", summary_path)
        return []

    class_names = summary.get("class_names")
    if isinstance(class_names, list) and all(isinstance(item, str) for item in class_names):
        return class_names
    return []


def _parse_class_labels(value: str | None) -> list[str] | None:
    if value is None:
        return DEFAULT_CLASS_LABELS
    labels = [label.strip() for label in value.split(",") if label.strip()]
    return labels or None


def find_prediction_csvs(runs_dir: Path, csv_name: str) -> list[Path]:
    csv_paths = sorted(runs_dir.rglob(csv_name))
    if not csv_paths:
        raise FileNotFoundError(f"No {csv_name!r} files found below {runs_dir}")
    return csv_paths


def read_predictions(
    csv_paths: list[Path],
    true_column: str,
    pred_column: str,
) -> tuple[list[tuple[Path, list[str], list[str]]], list[str]]:
    datasets = []
    labels = OrderedDict()

    for csv_path in csv_paths:
        y_true: list[str] = []
        y_pred: list[str] = []

        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"{csv_path} has no header row")
            missing_columns = [col for col in (true_column, pred_column) if col not in reader.fieldnames]
            if missing_columns:
                raise ValueError(f"{csv_path} is missing column(s): {', '.join(missing_columns)}")

            for row in reader:
                true_label = row[true_column]
                pred_label = row[pred_column]
                y_true.append(true_label)
                y_pred.append(pred_label)
                labels.setdefault(true_label, None)
                labels.setdefault(pred_label, None)

        if not y_true:
            logger.warning("Skipping empty predictions file: %s", csv_path)
            continue
        datasets.append((csv_path, y_true, y_pred))

    if not datasets:
        raise ValueError("No non-empty prediction CSV files were found")

    return datasets, list(labels.keys())


def flatten_predictions(
    datasets: list[tuple[Path, list[str], list[str]]],
) -> tuple[list[str], list[str]]:
    y_true_all: list[str] = []
    y_pred_all: list[str] = []
    for _, y_true, y_pred in datasets:
        y_true_all.extend(y_true)
        y_pred_all.extend(y_pred)
    return y_true_all, y_pred_all


def infer_class_labels(csv_paths: list[Path], observed_labels: list[str], explicit_labels: list[str] | None) -> list[str]:
    if explicit_labels:
        return explicit_labels

    for csv_path in csv_paths:
        summary_labels = _load_summary_class_names(csv_path)
        if summary_labels and set(observed_labels).issubset(set(summary_labels)):
            return summary_labels

    return observed_labels


def display_class_labels(class_labels: list[str], display_labels: dict[str, str]) -> list[str]:
    return [display_labels.get(label, label) for label in class_labels]


def mean_confusion_matrices(
    datasets: list[tuple[Path, list[str], list[str]]],
    class_labels: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    counts = []
    normalized = []

    for csv_path, y_true, y_pred in datasets:
        unknown = (set(y_true) | set(y_pred)) - set(class_labels)
        if unknown:
            raise ValueError(
                f"{csv_path} contains labels missing from --class-labels: {', '.join(sorted(unknown))}"
            )

        cm_counts = confusion_matrix(y_true, y_pred, labels=class_labels).astype(float)
        row_sums = cm_counts.sum(axis=1, keepdims=True)
        cm_normalized = np.divide(
            cm_counts,
            row_sums,
            out=np.zeros_like(cm_counts, dtype=float),
            where=row_sums != 0,
        )
        counts.append(cm_counts)
        normalized.append(cm_normalized)

    return np.mean(counts, axis=0), np.mean(normalized, axis=0)


def _format_annotations(
    mean_cm_counts: np.ndarray,
    mean_cm_normalized: np.ndarray,
    percent_decimals: int,
    count_decimals: int,
) -> np.ndarray:
    annotations = np.empty(mean_cm_normalized.shape, dtype=object)
    for i in range(mean_cm_normalized.shape[0]):
        for j in range(mean_cm_normalized.shape[1]):
            percent = mean_cm_normalized[i, j] * 100
            count = mean_cm_counts[i, j]
            annotations[i, j] = f"{percent:.{percent_decimals}f}%"#\n({count:.{count_decimals}f})"
    return annotations


def plot_mean_confusion_matrix(
    mean_cm_counts: np.ndarray,
    mean_cm_normalized: np.ndarray,
    class_labels: list[str],
    output_path: Path,
    title: str,
    x_label: str,
    y_label: str,
    colorbar_label: str,
    cmap: str,
    figsize: tuple[float, float],
    dpi: int,
    title_font_size: int,
    axis_label_font_size: int,
    tick_font_size: int,
    annot_font_size: int,
    colorbar_font_size: int,
    percent_decimals: int,
    count_decimals: int,
) -> None:
    annotations = _format_annotations(
        mean_cm_counts,
        mean_cm_normalized,
        percent_decimals=percent_decimals,
        count_decimals=count_decimals,
    )

    fig, ax = plt.subplots(figsize=figsize)
    heatmap = sns.heatmap(
        mean_cm_normalized,
        annot=annotations,
        annot_kws={"fontsize": annot_font_size},
        fmt="",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        square=True,
        linewidths=0.5,
        linecolor="white",
        xticklabels=class_labels,
        yticklabels=class_labels,
        cbar_kws={"label": colorbar_label},
        ax=ax,
    )

    ax.set_title(title, fontsize=title_font_size, pad=16)
    ax.set_xlabel(x_label, fontsize=axis_label_font_size, labelpad=10)
    ax.set_ylabel(y_label, fontsize=axis_label_font_size, labelpad=10)
    ax.tick_params(axis="both", labelsize=tick_font_size)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    colorbar = heatmap.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=colorbar_font_size)
    colorbar.ax.yaxis.label.set_size(colorbar_font_size)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def output_paths_for_formats(output_path: Path, output_formats: list[str] | None) -> list[Path]:
    if not output_formats:
        return [output_path]

    paths = []
    for output_format in output_formats:
        suffix = output_format.lower().lstrip(".")
        paths.append(output_path.with_suffix(f".{suffix}"))
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot a mean confusion matrix from predictions CSV files under a ResNet18 runs directory."
    )
    parser.add_argument("runs_dir", type=Path, help="Path to a resnet18_runs directory")
    parser.add_argument("--csv-name", default=DEFAULT_CSV_NAME, help="Prediction CSV filename to search for")
    parser.add_argument("--output", type=Path, default=None, help="Output image path")
    parser.add_argument(
        "--output-formats",
        nargs="+",
        default=DEFAULT_OUTPUT_FORMATS,
        help="Optional list of formats to save, e.g. png pdf svg. Uses --output stem.",
    )
    parser.add_argument("--true-column", default=DEFAULT_TRUE_COLUMN, help="Ground-truth label column name")
    parser.add_argument("--pred-column", default=DEFAULT_PRED_COLUMN, help="Predicted label column name")
    parser.add_argument("--class-labels", default=None, help="Comma-separated class labels in plotting order")
    parser.add_argument("--title", default=DEFAULT_TITLE, help="Plot title")
    parser.add_argument("--x-label", default=DEFAULT_X_LABEL, help="X-axis label")
    parser.add_argument("--y-label", default=DEFAULT_Y_LABEL, help="Y-axis label")
    parser.add_argument("--colorbar-label", default=DEFAULT_COLORBAR_LABEL, help="Colorbar label")
    parser.add_argument("--font-path", type=Path, default=DEFAULT_FONT_PATH, help="Path to CMU Serif .ttf")
    parser.add_argument("--cmap", default=DEFAULT_CMAP, help="Matplotlib colormap")
    parser.add_argument("--figsize", nargs=2, type=float, default=DEFAULT_FIGSIZE, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--title-font-size", type=int, default=DEFAULT_TITLE_FONT_SIZE)
    parser.add_argument("--axis-label-font-size", type=int, default=DEFAULT_AXIS_LABEL_FONT_SIZE)
    parser.add_argument("--tick-font-size", type=int, default=DEFAULT_TICK_FONT_SIZE)
    parser.add_argument("--annot-font-size", type=int, default=DEFAULT_ANNOT_FONT_SIZE)
    parser.add_argument("--colorbar-font-size", type=int, default=DEFAULT_COLORBAR_FONT_SIZE)
    parser.add_argument("--percent-decimals", type=int, default=DEFAULT_PERCENT_DECIMALS)
    parser.add_argument("--count-decimals", type=int, default=DEFAULT_COUNT_DECIMALS)
    parser.add_argument(
        "--save-arrays",
        action="store_true",
        help="Also save mean count and row-normalized arrays next to the plot",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args()

    runs_dir = args.runs_dir.resolve()
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")

    output_path = args.output
    if output_path is None:
        output_path = runs_dir.parent / DEFAULT_OUTPUT_NAME
    output_path = _resolve_path(output_path, base=Path.cwd())

    font_path = _resolve_path(args.font_path)
    font_name = _configure_font(font_path)

    csv_paths = find_prediction_csvs(runs_dir, args.csv_name)
    datasets, observed_labels = read_predictions(csv_paths, args.true_column, args.pred_column)
    class_labels = infer_class_labels(csv_paths, observed_labels, _parse_class_labels(args.class_labels))
    y_true_all, y_pred_all = flatten_predictions(datasets)
    mean_cm_counts, mean_cm_normalized = mean_confusion_matrices(datasets, class_labels)

    saved_paths = output_paths_for_formats(output_path, args.output_formats)
    for saved_path in saved_paths:
        plot_mean_confusion_matrix(
            mean_cm_counts=mean_cm_counts,
            mean_cm_normalized=mean_cm_normalized,
            class_labels=display_class_labels(class_labels, DEFAULT_DISPLAY_LABELS),
            output_path=saved_path,
            title=args.title,
            x_label=args.x_label,
            y_label=args.y_label,
            colorbar_label=args.colorbar_label,
            cmap=args.cmap,
            figsize=tuple(args.figsize),
            dpi=args.dpi,
            title_font_size=args.title_font_size,
            axis_label_font_size=args.axis_label_font_size,
            tick_font_size=args.tick_font_size,
            annot_font_size=args.annot_font_size,
            colorbar_font_size=args.colorbar_font_size,
            percent_decimals=args.percent_decimals,
            count_decimals=args.count_decimals,
        )

    if args.save_arrays:
        np.save(output_path.with_name(f"{output_path.stem}_counts.npy"), mean_cm_counts)
        np.save(output_path.with_name(f"{output_path.stem}_normalized.npy"), mean_cm_normalized)

    if y_true_all and y_pred_all:
        print("\nClassification report (all predictions):")
        print(
            classification_report(
                y_true_all,
                y_pred_all,
                labels=class_labels,
                target_names=display_class_labels(class_labels, DEFAULT_DISPLAY_LABELS),
                zero_division=0,
            )
        )
        accuracy = accuracy_score(y_true_all, y_pred_all)
        f1_macro = f1_score(y_true_all, y_pred_all, labels=class_labels, average="macro", zero_division=0)
        f1_weighted = f1_score(y_true_all, y_pred_all, labels=class_labels, average="weighted", zero_division=0)
        precision_macro = precision_score(
            y_true_all, y_pred_all, labels=class_labels, average="macro", zero_division=0
        )
        recall_macro = recall_score(
            y_true_all, y_pred_all, labels=class_labels, average="macro", zero_division=0
        )
        print(
            "Summary metrics: "
            f"accuracy={accuracy:.4f}, "
            f"f1_macro={f1_macro:.4f}, "
            f"f1_weighted={f1_weighted:.4f}, "
            f"precision_macro={precision_macro:.4f}, "
            f"recall_macro={recall_macro:.4f}"
        )

    logger.info("Used font: %s (%s)", font_name, font_path)
    logger.info("Found %d prediction CSV file(s)", len(csv_paths))
    for saved_path in saved_paths:
        logger.info("Saved mean confusion matrix to: %s", saved_path)


if __name__ == "__main__":
    main()
