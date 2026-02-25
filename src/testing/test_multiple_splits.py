import csv
import json
import re
import sys
import time
import os
import torch
from pathlib import Path
from argparse import ArgumentParser

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.pushover import send_notification
from evaluate_resnet18 import evaluate_resnet18
from resnet_dataloader import ResNetDataloader
from train_resnet18 import ResNet18Test
from multi_run_evaluation import multi_run_evaluation


def plan_pending_runs(config, runs_dir):
    """
    Build a list of ALL expected runs from the config, check the filesystem
    state of each, and return only the ones that still need work.

    Each returned task is a dict with:
        real_count, synthetic_count, run_number, run_dir,
        action   – "train" | "resume" | "evaluate"
        resume_checkpoint – path string (only when action == "resume")
    """
    real_image_counts = config["evaluation"]["splits"]["real_image_counts"]
    synthetic_image_counts = config["evaluation"]["splits"]["synthetic_image_counts"]
    splits = list(zip(real_image_counts, synthetic_image_counts))
    training_runs_per_split = config["evaluation"]["training_runs_per_split"]

    pending = []

    for real_count, synthetic_count in splits:
        for run_number in range(1, training_runs_per_split + 1):
            run_name = f"{real_count}-real_{synthetic_count}-synthetic_run{run_number}"
            run_dir = runs_dir / run_name

            # ---- Directory does not exist → full training needed ----
            if not run_dir.exists():
                pending.append(_task(real_count, synthetic_count, run_number,
                                     run_dir, action="train"))
                continue

            latest_ckpt = run_dir / "resnet18_latest.ckpt"
            final_ckpts = list(run_dir.glob("resnet18_final_*.ckpt"))
            eval_summary = run_dir / "evaluation" / "evaluation_summary_val.json"

            if latest_ckpt.exists():
                # resnet18_latest.ckpt is renamed to resnet18_final_* when
                # training finishes, so its presence means training was
                # interrupted → resume from this checkpoint.
                epoch = _read_epoch(latest_ckpt)
                print(f"[RESUME] {run_name} — interrupted training detected "
                      f"(resnet18_latest.ckpt, epoch {epoch})")
                pending.append(_task(real_count, synthetic_count, run_number,
                                     run_dir, action="resume",
                                     resume_checkpoint=str(latest_ckpt)))

            elif final_ckpts and not eval_summary.exists():
                # Training finished (final checkpoint exists) but evaluation
                # was not completed.
                print(f"[EVAL]   {run_name} — training complete, evaluation missing")
                pending.append(_task(real_count, synthetic_count, run_number,
                                     run_dir, action="evaluate"))

            elif eval_summary.exists():
                # Fully done — nothing to do.
                print(f"[DONE]   {run_name}")

            else:
                # Directory exists but no latest or final checkpoint.
                # Fall back to resnet18_lowest_val_loss.ckpt if available.
                lowest_loss_ckpt = run_dir / "resnet18_lowest_val_loss.ckpt"
                if lowest_loss_ckpt.exists():
                    epoch = _read_epoch(lowest_loss_ckpt)
                    print(f"[RESUME] {run_name} — no latest/final ckpt, "
                          f"falling back to resnet18_lowest_val_loss.ckpt (epoch {epoch})")
                    pending.append(_task(real_count, synthetic_count, run_number,
                                         run_dir, action="resume",
                                         resume_checkpoint=str(lowest_loss_ckpt)))
                else:
                    print(f"[TRAIN]  {run_name} — directory exists but no checkpoints found")
                    pending.append(_task(real_count, synthetic_count, run_number,
                                         run_dir, action="train"))

    return pending


def _read_epoch(ckpt_path):
    """Read the epoch number from a checkpoint file (CPU-only load)."""
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        return ckpt.get("epoch", "?")
    except Exception:
        return "?"


def _task(real_count, synthetic_count, run_number, run_dir,
          action, resume_checkpoint=None):
    """Helper to build a task dict."""
    return {
        "real_count": real_count,
        "synthetic_count": synthetic_count,
        "run_number": run_number,
        "run_dir": str(run_dir),
        "action": action,
        "resume_checkpoint": resume_checkpoint,
    }


# ---------------------------------------------------------------------------- #
#                          Results Logging Utilities                           #
# ---------------------------------------------------------------------------- #

def resolve_config_value(config, dotted_path):
    """Resolve a dotted path like 'data.resolution' from the config dict.

    Returns the value if found, or None if the path is invalid.
    """
    keys = dotted_path.split(".")
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return None
    return value


def is_synthetic_split_test(config):
    """Return True when the independent variable is the synthetic image count."""
    ind_var = config.get("evaluation", {}).get("independent_variable", "")
    return ind_var == "evaluation.splits.synthetic_image_count"


def get_independent_variable_value(config, real_count, synthetic_count):
    """Derive the independent-variable value for a specific split."""
    ind_var = config.get("evaluation", {}).get("independent_variable", "")
    if ind_var == "evaluation.splits.synthetic_image_count":
        return synthetic_count
    if ind_var == "evaluation.splits.real_image_count":
        return real_count
    value = resolve_config_value(config, ind_var)
    if value is None:
        print(f"[WARNING] Could not resolve independent variable "
              f"'{ind_var}' from config — using 'unknown_independent_variable'.")
        return "unknown_independent_variable"
    return value


def _pretty_label(s):
    """Turn a dotted config path or snake_case name into a Title Case label."""
    if not isinstance(s, str):
        return str(s)
    return s.rsplit(".", 1)[-1].replace("_", " ").title()


def init_results(config):
    """Build the initial results dict skeleton from the evaluation config."""
    evaluation = config.get("evaluation", {})
    eval_id = evaluation.get("evaluation_id", "unknown")
    ind_var = evaluation.get("independent_variable", "unknown")
    metrics = evaluation.get("dependent_variable", ["val_accuracy"])
    if isinstance(metrics, str):
        metrics = [metrics]

    x_label = _pretty_label(ind_var)
    y_label = _pretty_label(metrics[0]) if metrics else "Metric"

    return {
        "evaluation_id": eval_id,
        "independent_variable": ind_var,
        "independent_variable_value": None,  # set once per run
        "metrics": metrics,
        "splits": [],
        "plots": {
            "master_plot_config": True,
            "bar_chart": {
                "x_axis_label": x_label,
                "y_axis_label": y_label,
                "title": f"{y_label} vs. {x_label} for ResNet18",
            },
            "line_chart": {
                "x_axis_label": x_label,
                "y_axis_label": y_label,
                "title": f"{y_label} vs. {x_label} for ResNet18",
            },
        },
    }


def load_or_init_results(results_path, config):
    """Load an existing results.json or create a fresh skeleton."""
    if results_path.exists():
        with open(results_path, "r") as f:
            print(f"[INFO] Loaded existing results from {results_path}")
            return json.load(f)
    return init_results(config)


def find_or_create_split_entry(results, real_count, synthetic_count,
                               independent_value):
    """Return the matching split entry, creating one if it does not exist."""
    for entry in results["splits"]:
        if (entry.get("real_image_count") == real_count
                and entry.get("synthetic_image_count") == synthetic_count):
            return entry

    # ---- Create new entry ----
    entry = {
        "split_name": f"{real_count}-real_{synthetic_count}-synthetic",
        "real_image_count": real_count,
        "synthetic_image_count": synthetic_count,
        "runs": [],
    }
    results["splits"].append(entry)
    return entry


def extract_run_metrics(eval_summary_path, model_instance,
                        dependent_variables, run_dir=None):
    """Read dependent-variable metrics from the evaluation summary (and
    optionally from the training model / checkpoint for val_loss)."""
    _METRIC_MAP = {
        "val_accuracy": "accuracy",
        "f1_score": "f1_score",
        "precision": "precision",
        "recall": "recall",
    }

    eval_summary = {}
    if eval_summary_path.exists():
        with open(eval_summary_path, "r") as f:
            eval_summary = json.load(f)

    run_metrics = {}
    for dep_var in dependent_variables:
        if dep_var == "val_loss":
            # Prefer the live model attribute; fall back to the checkpoint.
            if model_instance is not None and hasattr(model_instance, "lowest_val_loss"):
                run_metrics[dep_var] = round(float(model_instance.lowest_val_loss), 6)
            elif run_dir is not None:
                ckpt_path = Path(run_dir) / "resnet18_lowest_val_loss.ckpt"
                if ckpt_path.exists():
                    ckpt = torch.load(str(ckpt_path), map_location="cpu",
                                      weights_only=False)
                    val_losses = ckpt.get("val_loss", [])
                    if val_losses:
                        run_metrics[dep_var] = round(float(min(val_losses)), 6)
        elif dep_var in _METRIC_MAP:
            key = _METRIC_MAP[dep_var]
            if key in eval_summary:
                run_metrics[dep_var] = round(float(eval_summary[key]), 6)
        else:
            # Unknown metric — try a literal lookup in the eval summary.
            if dep_var in eval_summary:
                try:
                    run_metrics[dep_var] = round(float(eval_summary[dep_var]), 6)
                except (TypeError, ValueError):
                    print(f"[WARNING] Could not convert '{dep_var}' value to float")
            else:
                print(f"[WARNING] Dependent variable '{dep_var}' not found "
                      f"in evaluation summary")
    return run_metrics


def save_results(results, results_path):
    """Atomically write the results dict to a JSON file."""
    tmp_path = results_path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(results, f, indent=4)
    tmp_path.rename(results_path)
    print(f"[INFO] Results saved to {results_path}")


def export_results_csv(results, csv_path):
    """Export the results to a flat CSV (one row per run)."""
    dep_vars = results.get("metrics", [])
    ind_var = results.get("independent_variable", "unknown")
    eval_id = results.get("evaluation_id", "unknown")

    rows = []
    for entry in results.get("splits", []):
        real_count = entry.get("real_image_count", "")
        synthetic_count = entry.get("synthetic_image_count", "")
        ind_value = entry.get("independent_variable_value", "")

        for run_idx, run in enumerate(entry.get("runs", []), start=1):
            row = {
                "evaluation_id": eval_id,
                "independent_variable": ind_var,
                "independent_variable_value": ind_value,
                "real_image_count": real_count,
                "synthetic_image_count": synthetic_count,
                "run_number": run_idx,
            }
            for dv in dep_vars:
                row[dv] = run.get(dv, "")
            rows.append(row)

    if not rows:
        print("[WARNING] No data to export to CSV")
        return

    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] CSV exported to {csv_path}")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Train multiple resnet18 models on different splits "
                    "of the dataset and evaluate their performance."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="/home/ap/cloud/Master/aris_master/src/testing/testing_config.json",
        help="Path to config JSON file",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train models for each split",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate models for each split",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Additionally export results to a .csv file",
    )
    args = parser.parse_args()
    config_path = args.config

    # Load config
    with open(config_path, "r") as f:
        config = json.load(f)

    # Make 'runs' folder
    runs_dir = Path(config["logging"]["output_dir"]) / "resnet18_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # ---- Initialize results tracking ----
    results_path = Path(config["logging"]["output_dir"]) / "results.json"
    results = load_or_init_results(results_path, config)
    dependent_variables = config.get("evaluation", {}).get(
        "dependent_variable", ["val_accuracy"])
    if isinstance(dependent_variables, str):
        dependent_variables = [dependent_variables]

    # Store independent variable value at top level if already known
    ind_val_first = get_independent_variable_value(config, 
        config["evaluation"]["splits"]["real_image_counts"][0],
        config["evaluation"]["splits"]["synthetic_image_counts"][0])
    if results.get("independent_variable_value") is None:
        results["independent_variable_value"] = ind_val_first

    # Pre-populate split entries so independent variable values are logged
    # as soon as the script starts (before any training begins).
    real_image_counts = config["evaluation"]["splits"]["real_image_counts"]
    synthetic_image_counts = config["evaluation"]["splits"]["synthetic_image_counts"]
    splits = list(zip(real_image_counts, synthetic_image_counts))
    for rc, sc in splits:
        ind_val = get_independent_variable_value(config, rc, sc)
        find_or_create_split_entry(results, rc, sc, ind_val)
    save_results(results, results_path)

    # ---- Plan work: figure out what's already done vs. what's pending ----
    pending_tasks = plan_pending_runs(config, runs_dir)

    if not pending_tasks:
        print("\n[INFO] All runs already completed — nothing to do.")
    else:
        # Print a summary of planned work
        print(f"\n[INFO] {len(pending_tasks)} pending task(s):")
        for t in pending_tasks:
            tag = t["action"].upper()
            name = (f"{t['real_count']}-real_{t['synthetic_count']}-synthetic"
                    f"_run{t['run_number']}")
            print(f"  [{tag:8s}] {name}")
        print()

    # ---- Execute pending tasks ----
    start_time = time.time()
    total_tasks = len(pending_tasks)

    for task_idx, task in enumerate(pending_tasks, start=1):
        real_count = task["real_count"]
        synthetic_count = task["synthetic_count"]
        run_number = task["run_number"]
        action = task["action"]
        run_dir = task["run_dir"]

        print(f"\n{'='*70}")
        print(f"[INFO] Task {task_idx}/{total_tasks}: "
              f"{real_count}-real_{synthetic_count}-synthetic "
              f"run{run_number}  ({action})")
        print(f"{'='*70}")

        # -- Training (fresh or resumed) --
        model_instance = None
        if action in ("train", "resume"):
            if not args.train and action == "train":
                print("[SKIP] --train not set, skipping fresh training")
            else:
                print(f"[INFO] {'Resuming' if action == 'resume' else 'Starting'} "
                      f"training on {real_count} real + {synthetic_count} synthetic images…")
                dataloader = ResNetDataloader(
                    config_path,
                    real_image_count=real_count,
                    synthetic_image_count=synthetic_count,
                )
                model = ResNet18Test(
                    config_path,
                    resnet_dataloader=dataloader,
                    output_dir=run_dir,
                    resume_checkpoint_path=task["resume_checkpoint"],
                )
                model.train()
                model_instance = model
                # After training completes the run also needs evaluation,
                # so fall through to the evaluate block below.
                action = "evaluate"

        # -- Evaluation --
        if action == "evaluate":
            if not args.evaluate:
                print("[SKIP] --evaluate not set, skipping evaluation")
            else:
                print(f"[INFO] Evaluating {real_count}-real "
                      f"{synthetic_count}-synthetic run{run_number}…")
                dataloader = ResNetDataloader(
                    config_path,
                    real_image_count=real_count,
                    synthetic_image_count=synthetic_count,
                )
                evaluate_resnet18(
                    dataloader,
                    checkpoint_dir=run_dir,
                    best_checkpoint="lowest_val_loss",
                )

                # ---- Log dependent-variable metrics to results.json ----
                eval_summary_path = (Path(run_dir) / "evaluation"
                                     / "evaluation_summary_val.json")
                run_metrics = extract_run_metrics(
                    eval_summary_path, model_instance,
                    dependent_variables, run_dir,
                )
                run_metrics["run_id"] = f"run{run_number}"
                ind_value = get_independent_variable_value(
                    config, real_count, synthetic_count,
                )
                # Store independent variable value at top level (once)
                if results.get("independent_variable_value") is None:
                    results["independent_variable_value"] = ind_value
                split_entry = find_or_create_split_entry(
                    results, real_count, synthetic_count,
                    ind_value,
                )
                split_entry["runs"].append(run_metrics)
                save_results(results, results_path)

        # -------------------- Estimate remaining time -------------------- #
        elapsed = time.time() - start_time
        avg = elapsed / task_idx
        remaining = avg * (total_tasks - task_idx)
        remaining_hms = time.strftime("%H:%M:%S", time.gmtime(remaining))
        eta = time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() + remaining))
        print(f"[INFO] {task_idx}/{total_tasks} done  |  "
              f"elapsed {elapsed:.0f}s  |  "
              f"remaining ~{remaining_hms}  |  ETA {eta}")

    print(f"\n[INFO] All training and evaluation complete!")

    # ---- Optional CSV export ----
    if args.csv:
        csv_path = Path(config["logging"]["output_dir"]) / "results.csv"
        export_results_csv(results, csv_path)

    multi_run_evaluation(resnet18_runs_dir=str(runs_dir))