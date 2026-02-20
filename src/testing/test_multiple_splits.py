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
    args = parser.parse_args()
    config_path = args.config

    # Load config
    with open(config_path, "r") as f:
        config = json.load(f)

    # Make 'runs' folder
    runs_dir = Path(config["logging"]["output_dir"]) / "resnet18_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

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
    multi_run_evaluation(resnet18_runs_dir=str(runs_dir))