#!/usr/bin/env python3
"""Reevaluate ResNet18 testing results without mutating originals.

Creates a parallel results tree under <testing-dir>/reevaluated_results.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision
from sklearn.metrics import accuracy_score, f1_score


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunMetrics:
    split_name: str
    run_id: str
    accuracy: float
    macro_f1: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reevaluate results.json under a testing directory by computing macro F1 from "
            "predictions_test.csv and/or reevaluating checkpoints."
        )
    )
    parser.add_argument(
        "--testing-dir",
        type=Path,
        required=True,
        help="Root testing directory to scan for results.json files.",
    )
    parser.add_argument(
        "--macro-f1",
        action="store_true",
        help="Append macro F1 scores computed from predictions_test.csv files.",
    )
    parser.add_argument(
        "--keep-evaluation-id",
        action="store_true",
        help="Keep the original evaluation_id in the reevaluated results.json.",
    )
    parser.add_argument(
        "--reevaluate-checkpoints",
        action="store_true",
        help="Run inference with resnet18_best_val_acc.ckpt and append val/test metrics.",
    )
    parser.add_argument(
        "--reevaluate-lowest-val",
        "--revaluate-lowest-val",
        "--revaluate_lowest_val",
        dest="reevaluate_lowest_val",
        action="store_true",
        help="Reevaluate using resnet18_lowest_val_loss.ckpt instead of best-val-acc.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Rerun checkpoint inference even when reevaluated val/test prediction CSVs "
            "already exist."
        ),
    )
    return parser.parse_args()


def discover_results_files(testing_dir: Path) -> list[Path]:
    if not testing_dir.exists():
        raise FileNotFoundError(f"Testing directory not found: {testing_dir}")

    results_paths: list[Path] = []
    for path in testing_dir.rglob("results.json"):
        if "reevaluated_results" in path.parts:
            continue
        results_paths.append(path)

    return sorted(results_paths)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4)
    logger.info("Saved results JSON: %s", path)


def get_new_evaluation_id(original: str, keep: bool) -> str:
    if keep:
        return original
    if original.endswith("_reevaluated"):
        return original
    return f"{original}_reevaluated"


def find_resnet18_runs_dir(results_path: Path) -> Optional[Path]:
    candidates = [
        results_path.parent / "resnet18_runs",
        results_path.parent / "resnet_runs",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    for path in results_path.parent.rglob("resnet18_runs"):
        return path

    return None


def parse_run_id(path: Path) -> Optional[str]:
    match = re.search(r"run[_-]?(\d+)", path.as_posix())
    if not match:
        return None
    return f"run{int(match.group(1))}"


def find_config_path(run_dir: Path, result_dir: Path) -> Optional[Path]:
    search_dirs = [run_dir, result_dir]
    direct_names = ["config.json", "training_config.json", "resnet_config.json"]

    for search_dir in search_dirs:
        for name in direct_names:
            candidate = search_dir / name
            if candidate.exists():
                return candidate

    json_candidates = []
    for search_dir in search_dirs:
        for path in search_dir.glob("*.json"):
            name = path.name
            if name == "results.json":
                continue
            if name.startswith("evaluation_summary_"):
                continue
            json_candidates.append(path)

    config_candidates = [p for p in json_candidates if "config" in p.name]
    if config_candidates:
        return sorted(config_candidates)[0]

    return None


def find_checkpoint_path(run_dir: Path, use_lowest: bool) -> Optional[Path]:
    ckpt_name = "resnet18_lowest_val_loss.ckpt" if use_lowest else "resnet18_best_val_acc.ckpt"
    for path in run_dir.rglob(ckpt_name):
        return path
    return None


def parse_split_counts(split_name: str) -> tuple[Optional[int], Optional[int]]:
    match = re.search(r"(\d+)-real_(\d+)-synthetic", split_name)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def resolve_real_image_count(config_path: Path, split_name: str) -> Optional[int]:
    config = load_json(config_path)
    data_config = config.get("data", {})
    data_real_image_count = data_config.get("real_image_count")
    if data_real_image_count is not None:
        return int(data_real_image_count)

    evaluation_splits = config.get("evaluation", {}).get("splits", {})
    real_counts = evaluation_splits.get("real_image_counts", [])
    synthetic_counts = evaluation_splits.get("synthetic_image_counts", [])
    split_real_count, split_synthetic_count = parse_split_counts(split_name)

    if split_real_count is None:
        if len(real_counts) == 1:
            return int(real_counts[0])
        return None

    for idx, real_count in enumerate(real_counts):
        synthetic_count = synthetic_counts[idx] if idx < len(synthetic_counts) else None
        if int(real_count) == split_real_count and (
            split_synthetic_count is None
            or synthetic_count is None
            or int(synthetic_count) == split_synthetic_count
        ):
            return int(real_count)

    return split_real_count


def discover_run_dirs(resnet_runs_dir: Path, use_lowest: bool) -> list[Path]:
    ckpt_name = "resnet18_lowest_val_loss.ckpt" if use_lowest else "resnet18_best_val_acc.ckpt"
    run_dirs = {path.parent for path in resnet_runs_dir.rglob(ckpt_name)}
    if run_dirs:
        return sorted(run_dirs)
    return sorted([p for p in resnet_runs_dir.iterdir() if p.is_dir()])


def resolve_split_name(split_names: list[str], path: Path) -> Optional[str]:
    for split_name in split_names:
        if split_name in path.as_posix():
            return split_name
    if len(split_names) == 1:
        return split_names[0]
    return None


def read_predictions_csv(csv_path: Path) -> tuple[list[str], list[str]]:
    logger.info("Reading predictions CSV: %s", csv_path)
    true_labels: list[str] = []
    pred_labels: list[str] = []

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "true_label" not in reader.fieldnames or "predicted_label" not in reader.fieldnames:
            raise ValueError(f"Missing label columns in {csv_path}")
        for row in reader:
            true_labels.append(row["true_label"])
            pred_labels.append(row["predicted_label"])

    return true_labels, pred_labels


def compute_macro_f1_from_predictions(csv_path: Path) -> float:
    true_labels, pred_labels = read_predictions_csv(csv_path)
    if not true_labels:
        raise ValueError(f"No samples found in {csv_path}")
    macro_f1 = float(f1_score(true_labels, pred_labels, average="macro"))
    logger.info("Computed macro F1 %.6f from %d rows in %s", macro_f1, len(true_labels), csv_path)
    return macro_f1


def compute_accuracy_macro_f1_from_predictions(csv_path: Path) -> tuple[float, float]:
    true_labels, pred_labels = read_predictions_csv(csv_path)
    if not true_labels:
        raise ValueError(f"No samples found in {csv_path}")
    accuracy = float(accuracy_score(true_labels, pred_labels))
    macro_f1 = float(f1_score(true_labels, pred_labels, average="macro"))
    logger.info(
        "Computed resumed metrics from %s: accuracy=%.6f macro_f1=%.6f rows=%d",
        csv_path,
        accuracy,
        macro_f1,
        len(true_labels),
    )
    return accuracy, macro_f1


def load_existing_run_metrics(
    output_eval_dir: Path,
    split_name: str,
    run_id: str,
) -> Optional[tuple[RunMetrics, RunMetrics]]:
    val_predictions_path = output_eval_dir / "val_predictions.csv"
    test_predictions_path = output_eval_dir / "test_predictions.csv"

    if not val_predictions_path.exists() and not test_predictions_path.exists():
        return None

    if not val_predictions_path.exists() or not test_predictions_path.exists():
        logger.info(
            "Found partial existing reevaluation output for split=%s run=%s at %s; rerunning inference",
            split_name,
            run_id,
            output_eval_dir,
        )
        return None

    logger.info(
        "Found complete existing reevaluation output for split=%s run=%s; loading metrics from CSVs",
        split_name,
        run_id,
    )
    val_accuracy, val_macro_f1 = compute_accuracy_macro_f1_from_predictions(val_predictions_path)
    test_accuracy, test_macro_f1 = compute_accuracy_macro_f1_from_predictions(test_predictions_path)

    return (
        RunMetrics(
            split_name=split_name,
            run_id=run_id,
            accuracy=test_accuracy,
            macro_f1=test_macro_f1,
        ),
        RunMetrics(
            split_name=split_name,
            run_id=run_id,
            accuracy=val_accuracy,
            macro_f1=val_macro_f1,
        ),
    )


def update_results_with_macro_f1(
    results: dict,
    macro_f1_by_split_run: dict[tuple[str, str], float],
) -> None:
    if not macro_f1_by_split_run:
        logger.info("No macro F1 values were matched into results.json")
        return

    metrics_list = results.get("metrics", [])
    if "test_f1_macro" not in metrics_list:
        metrics_list.append("test_f1_macro")
    results["metrics"] = metrics_list

    for split in results.get("splits", []):
        split_name = split.get("split_name")
        for run in split.get("runs", []):
            run_id = run.get("run_id")
            key = (split_name, run_id)
            if key in macro_f1_by_split_run:
                run["test_f1_macro"] = round(float(macro_f1_by_split_run[key]), 6)
                logger.info(
                    "Updated macro F1 for split=%s run=%s value=%.6f",
                    split_name,
                    run_id,
                    macro_f1_by_split_run[key],
                )


def save_predictions_csv(
    output_path: Path,
    class_names: list[str],
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    probabilities: np.ndarray,
    filepaths: list[str],
) -> None:
    df_rows = []
    for idx, (true_idx, pred_idx) in enumerate(zip(true_labels, pred_labels)):
        row = {
            "true_label": class_names[int(true_idx)],
            "predicted_label": class_names[int(pred_idx)],
            "correct": int(true_idx == pred_idx),
            "probabilities": probabilities[idx].tolist(),
        }
        for cls_idx, class_name in enumerate(class_names):
            row[f"prob_{class_name}"] = float(probabilities[idx, cls_idx])
        if filepaths:
            row["filepath"] = filepaths[idx]
        df_rows.append(row)

    if not df_rows:
        logger.info("No prediction rows to save for %s", output_path)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=df_rows[0].keys())
        writer.writeheader()
        writer.writerows(df_rows)
    logger.info("Saved %d prediction rows to %s", len(df_rows), output_path)


def build_model(num_classes: int, checkpoint_path: Path, device: torch.device) -> nn.Module:
    logger.info(
        "Building ResNet18 model with %d classes from checkpoint %s on %s",
        num_classes,
        checkpoint_path,
        device,
    )
    model = torchvision.models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    logger.info("Loaded checkpoint and set model to eval mode")

    return model


def evaluate_loader(
    model: nn.Module,
    dataloader,
    device: torch.device,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    dataset_size = len(dataloader.dataset) if hasattr(dataloader, "dataset") else "unknown"
    logger.info("Starting %s inference over %s samples", split_name, dataset_size)
    predictions: list[int] = []
    labels: list[int] = []
    probs: list[np.ndarray] = []
    filepaths: list[str] = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["image"].to(device)
            batch_labels = batch["class"].to(device)
            outputs = model(inputs)
            batch_probs = torch.softmax(outputs, dim=1)
            batch_preds = torch.argmax(outputs, dim=1)

            predictions.extend(batch_preds.cpu().numpy().tolist())
            labels.extend(batch_labels.cpu().numpy().tolist())
            probs.extend(batch_probs.cpu().numpy())

            if "filepath" in batch:
                filepaths.extend(batch["filepath"])

    logger.info("Finished %s inference with %d predictions", split_name, len(predictions))
    return (
        np.array(predictions),
        np.array(labels),
        np.array(probs),
        filepaths,
    )


def compute_metrics_from_predictions(labels: np.ndarray, preds: np.ndarray) -> tuple[float, float]:
    accuracy = float(accuracy_score(labels, preds))
    macro_f1 = float(f1_score(labels, preds, average="macro"))
    logger.info("Computed metrics: accuracy=%.6f macro_f1=%.6f samples=%d", accuracy, macro_f1, len(labels))
    return accuracy, macro_f1


def compute_run_metrics(
    run_dir: Path,
    result_dir: Path,
    split_name: str,
    output_eval_dir: Path,
    use_lowest: bool,
    force: bool,
) -> Optional[tuple[RunMetrics, RunMetrics]]:
    logger.info("Starting checkpoint reevaluation for run=%s split=%s", run_dir, split_name)
    run_id = parse_run_id(run_dir) or run_dir.name

    if not force:
        existing_metrics = load_existing_run_metrics(output_eval_dir, split_name, run_id)
        if existing_metrics is not None:
            logger.info("Skipping inference for split=%s run=%s because existing CSVs were reused", split_name, run_id)
            return existing_metrics
    else:
        logger.info("Force enabled; existing CSVs will be ignored for split=%s run=%s", split_name, run_id)

    config_path = find_config_path(run_dir, result_dir)
    if config_path is None:
        logger.warning("Skipping %s (no config json found)", run_dir)
        return None
    logger.info("Using config: %s", config_path)

    checkpoint_path = find_checkpoint_path(run_dir, use_lowest)
    if checkpoint_path is None:
        logger.warning("Skipping %s (checkpoint not found)", run_dir)
        return None
    logger.info("Using checkpoint: %s", checkpoint_path)

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src" / "testing"))
    from resnet_dataloader import ResNetDataloader

    real_image_count = resolve_real_image_count(config_path, split_name)
    if real_image_count is None:
        logger.warning("Skipping %s (real image count unresolved for split %s)", run_dir, split_name)
        return None
    logger.info("Resolved real_image_count=%d for split=%s", real_image_count, split_name)

    run_number_match = re.search(r"run[_-]?(\d+)", run_dir.as_posix())
    run_number = int(run_number_match.group(1)) if run_number_match else None
    logger.info("Resolved run_number=%s for %s", run_number, run_dir)

    logger.info("Creating ResNetDataloader with synthetic_image_count=0, preview=False, num_workers=0")
    dataloader = ResNetDataloader(
        config_path=str(config_path),
        real_image_count=real_image_count,
        synthetic_image_count=0,
        run_number=run_number,
        preview=False,
        num_workers=0,
    )

    if dataloader.val_loader is None or dataloader.test_loader is None:
        logger.warning("Skipping %s (val/test loader missing)", run_dir)
        return None
    logger.info(
        "Dataloaders ready: val_samples=%d test_samples=%d",
        len(dataloader.val_loader.dataset),
        len(dataloader.test_loader.dataset),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(dataloader.num_classes, checkpoint_path, device)

    val_preds, val_labels, val_probs, val_filepaths = evaluate_loader(
        model, dataloader.val_loader, device, "val"
    )
    test_preds, test_labels, test_probs, test_filepaths = evaluate_loader(
        model, dataloader.test_loader, device, "test"
    )

    logger.info("Computing validation metrics for %s", run_dir)
    val_accuracy, val_macro_f1 = compute_metrics_from_predictions(val_labels, val_preds)
    logger.info("Computing test metrics for %s", run_dir)
    test_accuracy, test_macro_f1 = compute_metrics_from_predictions(test_labels, test_preds)

    save_predictions_csv(
        output_eval_dir / "val_predictions.csv",
        dataloader.classes,
        val_labels,
        val_preds,
        val_probs,
        val_filepaths,
    )
    save_predictions_csv(
        output_eval_dir / "test_predictions.csv",
        dataloader.classes,
        test_labels,
        test_preds,
        test_probs,
        test_filepaths,
    )

    logger.info(
        "Finished run=%s split=%s test_accuracy=%.6f test_macro_f1=%.6f val_accuracy=%.6f val_macro_f1=%.6f",
        run_id,
        split_name,
        test_accuracy,
        test_macro_f1,
        val_accuracy,
        val_macro_f1,
    )

    return (
        RunMetrics(
            split_name=split_name,
            run_id=run_id,
            accuracy=test_accuracy,
            macro_f1=test_macro_f1,
        ),
        RunMetrics(
            split_name=split_name,
            run_id=run_id,
            accuracy=val_accuracy,
            macro_f1=val_macro_f1,
        ),
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    should_reevaluate_checkpoints = args.reevaluate_checkpoints or args.reevaluate_lowest_val
    logger.info("Starting reevaluation")
    logger.info("testing_dir=%s", args.testing_dir)
    logger.info(
        "Modes: macro_f1=%s reevaluate_checkpoints=%s reevaluate_lowest_val=%s keep_evaluation_id=%s force=%s",
        args.macro_f1,
        should_reevaluate_checkpoints,
        args.reevaluate_lowest_val,
        args.keep_evaluation_id,
        args.force,
    )

    if not args.macro_f1 and not should_reevaluate_checkpoints:
        logger.warning(
            "No reevaluation mode selected. Use --macro-f1, --reevaluate-checkpoints, "
            "or --reevaluate-lowest-val."
        )

    logger.info("Discovering results.json files")
    results_files = discover_results_files(args.testing_dir)
    if not results_files:
        logger.warning("No results.json files found under %s", args.testing_dir)
        return
    logger.info("Found %d results.json file(s)", len(results_files))

    for result_idx, results_path in enumerate(results_files, start=1):
        logger.info("Processing results file %d/%d: %s", result_idx, len(results_files), results_path)
        original = load_json(results_path)
        new_results = deepcopy(original)

        original_eval_id = original.get("evaluation_id", "unknown")
        new_eval_id = get_new_evaluation_id(original_eval_id, args.keep_evaluation_id)
        reevaluated_eval_id = get_new_evaluation_id(original_eval_id, keep=False)
        new_results["evaluation_id"] = new_eval_id
        logger.info(
            "Evaluation IDs: original=%s output=%s reevaluation_folder=%s",
            original_eval_id,
            new_eval_id,
            reevaluated_eval_id,
        )

        rel_parent = results_path.parent.relative_to(args.testing_dir)
        output_base = args.testing_dir / "reevaluated_results" / rel_parent
        output_results_path = output_base / "results.json"
        logger.info("Output results path: %s", output_results_path)

        split_names = [split.get("split_name") for split in original.get("splits", [])]
        logger.info("Discovered %d split(s): %s", len(split_names), ", ".join(str(s) for s in split_names))
        resnet_runs_dir = find_resnet18_runs_dir(results_path)
        source_resnet_runs_dir = str(resnet_runs_dir.resolve()) if resnet_runs_dir else None
        if resnet_runs_dir is None:
            logger.info("No local resnet18_runs directory found for %s", results_path)
        else:
            logger.info("Using resnet runs directory: %s", resnet_runs_dir)

        new_results["reevaluation_metadata"] = {
            "aggregated_evaluation_id": original_eval_id,
            "source_results_json": str(results_path.resolve()),
            "source_resnet18_runs_dir": source_resnet_runs_dir,
            "reevaluated_evaluation_id": reevaluated_eval_id,
            "reevaluated_results_json": str(output_results_path.resolve()),
        }

        macro_f1_by_split_run: dict[tuple[str, str], float] = {}
        if args.macro_f1:
            logger.info("Starting macro-F1 update from existing predictions_test.csv files")
            if resnet_runs_dir is None:
                logger.warning("No resnet18_runs dir found for %s", results_path)
            else:
                prediction_paths = sorted(resnet_runs_dir.rglob("predictions_test.csv"))
                logger.info("Found %d predictions_test.csv file(s)", len(prediction_paths))
                for pred_idx, pred_path in enumerate(prediction_paths, start=1):
                    logger.info("Processing predictions file %d/%d: %s", pred_idx, len(prediction_paths), pred_path)
                    split_name = resolve_split_name(split_names, pred_path)
                    if split_name is None:
                        logger.warning("Skipping %s (split name unresolved)", pred_path)
                        continue

                    run_id = parse_run_id(pred_path) or pred_path.parent.name
                    logger.info("Resolved predictions file to split=%s run=%s", split_name, run_id)
                    try:
                        macro_f1 = compute_macro_f1_from_predictions(pred_path)
                    except Exception as exc:
                        logger.warning("Failed macro F1 for %s: %s", pred_path, exc)
                        continue
                    macro_f1_by_split_run[(split_name, run_id)] = macro_f1

            update_results_with_macro_f1(new_results, macro_f1_by_split_run)
            logger.info("Finished macro-F1 update with %d matched value(s)", len(macro_f1_by_split_run))

        if not should_reevaluate_checkpoints:
            logger.info("Checkpoint reevaluation disabled; writing macro/copy results only")
            dump_json(output_results_path, new_results)
            logger.info("Wrote reevaluated results to %s", output_results_path)
            continue

        if resnet_runs_dir is None:
            logger.warning("No resnet18_runs dir found for %s", results_path)
            dump_json(output_results_path, new_results)
            continue

        reevaluated_eval_root = output_base / reevaluated_eval_id / "resnet18_runs"
        logger.info("Checkpoint reevaluation outputs will be written under: %s", reevaluated_eval_root)

        test_metrics_by_split_run: dict[tuple[str, str], RunMetrics] = {}
        val_metrics_by_split_run: dict[tuple[str, str], RunMetrics] = {}

        run_dirs = discover_run_dirs(resnet_runs_dir, args.reevaluate_lowest_val)
        checkpoint_name = "resnet18_lowest_val_loss.ckpt" if args.reevaluate_lowest_val else "resnet18_best_val_acc.ckpt"
        logger.info("Discovered %d run directories for checkpoint %s", len(run_dirs), checkpoint_name)

        for run_idx, run_dir in enumerate(run_dirs, start=1):
            logger.info("Processing run directory %d/%d: %s", run_idx, len(run_dirs), run_dir)
            split_name = resolve_split_name(split_names, run_dir)
            if split_name is None:
                if len(split_names) == 1:
                    split_name = split_names[0]
                    logger.info("Using only available split=%s for %s", split_name, run_dir)
                else:
                    logger.warning("Skipping %s (split name unresolved)", run_dir)
                    continue
            else:
                logger.info("Resolved run directory to split=%s", split_name)

            relative_run = run_dir
            try:
                relative_run = run_dir.relative_to(resnet_runs_dir)
            except ValueError:
                pass

            output_eval_dir = reevaluated_eval_root / relative_run / "evaluation"
            logger.info("Run output evaluation directory: %s", output_eval_dir)
            try:
                run_metrics = compute_run_metrics(
                    run_dir,
                    results_path.parent,
                    split_name,
                    output_eval_dir,
                    args.reevaluate_lowest_val,
                    args.force,
                )
            except Exception as exc:
                logger.warning("Failed reevaluation for %s: %s", run_dir, exc)
                continue

            if run_metrics is None:
                continue

            test_metrics, val_metrics = run_metrics

            test_metrics_by_split_run[(test_metrics.split_name, test_metrics.run_id)] = test_metrics
            val_metrics_by_split_run[(val_metrics.split_name, val_metrics.run_id)] = val_metrics
            logger.info("Stored reevaluated metrics for split=%s run=%s", test_metrics.split_name, test_metrics.run_id)

        metrics_list = new_results.get("metrics", [])
        for metric in ["val_accuracy", "val_f1_macro", "test_accuracy", "test_f1_macro"]:
            if metric not in metrics_list:
                metrics_list.append(metric)
        new_results["metrics"] = metrics_list
        logger.info(
            "Applying checkpoint metrics to results.json: test_matches=%d val_matches=%d",
            len(test_metrics_by_split_run),
            len(val_metrics_by_split_run),
        )

        for split in new_results.get("splits", []):
            split_name = split.get("split_name")
            for run in split.get("runs", []):
                run_id = run.get("run_id")
                key = (split_name, run_id)
                if key in test_metrics_by_split_run:
                    test_metrics = test_metrics_by_split_run[key]
                    run["test_accuracy"] = round(test_metrics.accuracy, 6)
                    run["test_f1_macro"] = round(test_metrics.macro_f1, 6)
                    logger.info(
                        "Updated test metrics for split=%s run=%s accuracy=%.6f macro_f1=%.6f",
                        split_name,
                        run_id,
                        test_metrics.accuracy,
                        test_metrics.macro_f1,
                    )

                if key in val_metrics_by_split_run:
                    val_metrics = val_metrics_by_split_run[key]
                    run["val_accuracy"] = round(val_metrics.accuracy, 6)
                    run["val_f1_macro"] = round(val_metrics.macro_f1, 6)
                    logger.info(
                        "Updated validation metrics for split=%s run=%s accuracy=%.6f macro_f1=%.6f",
                        split_name,
                        run_id,
                        val_metrics.accuracy,
                        val_metrics.macro_f1,
                    )

        dump_json(output_results_path, new_results)
        logger.info("Wrote reevaluated results to %s", output_results_path)

    logger.info("Finished reevaluation")


if __name__ == "__main__":
    main()
