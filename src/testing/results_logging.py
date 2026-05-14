import json
import csv
from pathlib import Path
import torch

import logging
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------- #
#                          Results Logger                                      #
# ---------------------------------------------------------------------------- #


class ResultsLogger:
    """Encapsulates results tracking for multi-split ResNet18 evaluations.

    Handles initialisation, persistence (JSON / CSV), and per-run metric
    logging so that callers only need a single object.
    """

    _METRIC_MAP = {
        # Legacy, unprefixed metric names map to values in val summary.
        "accuracy": "accuracy",
        "f1_score": "f1_score",
        "f1_macro": "f1_macro",
        "precision": "precision",
        "recall": "recall",
    }

    _ALWAYS_TRACKED_F1_METRICS = [
        "val_f1_score",
        "test_f1_score",
        "val_f1_macro",
        "test_f1_macro",
    ]

    _CHECKPOINT_FILES = {
        "lowest_val_loss": "resnet18_lowest_val_loss.ckpt",
        "val_loss": "resnet18_lowest_val_loss.ckpt",
        "highest_val_acc": "resnet18_best_val_acc.ckpt",
        "val_acc": "resnet18_best_val_acc.ckpt",
        "f1_macro": "resnet18_best_val_f1_macro.ckpt",
        "val_f1_macro": "resnet18_best_val_f1_macro.ckpt",
        "best_val_f1_macro": "resnet18_best_val_f1_macro.ckpt",
    }

    def __init__(self, config, output_dir=None):
        self.config = config
        evaluation = config.get("evaluation", {})

        # Determine output directory
        if output_dir is None:
            output_dir = config["logging"]["output_dir"]
        if not Path(output_dir).exists():
            # Make dir
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            
        self.results_path = Path(output_dir) / "results.json"

        # Dependent variables
        self.dependent_variables = evaluation.get(
            "metrics", ["val_accuracy"]
        )
        logger.debug(f"Initial dependent variables from config: {self.dependent_variables}")

        
        if isinstance(self.dependent_variables, str):
            self.dependent_variables = [self.dependent_variables]
        self._ensure_always_tracked_f1_metrics()

        # Load or initialise results
        self.results = self._load_or_init_results()
        self._ensure_results_metric_list()

        # Pre-populate split entries
        self._prepopulate_splits()

    # ------------------------------------------------------------------ #
    #  Static / private helpers                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_config_value(config, dotted_path):
        """Resolve a dotted path like 'data.resolution' from a config dict."""
        keys = dotted_path.split(".")
        value = config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        logger.debug(f"Resolving config value for path '{dotted_path}'")
        logger.debug(f"Resolved value: {value}")

        return value

    @staticmethod
    def _pretty_label(s):
        """Turn a dotted config path or snake_case name into Title Case."""
        if not isinstance(s, str):
            return str(s)
        return s.rsplit(".", 1)[-1].replace("_", " ").title()

    def _ensure_always_tracked_f1_metrics(self):
        """Always track weighted and macro F1 for validation and test."""
        for metric in self._ALWAYS_TRACKED_F1_METRICS:
            if metric not in self.dependent_variables:
                self.dependent_variables.append(metric)

    def _ensure_results_metric_list(self):
        """Keep existing results.json metric metadata in sync with logging."""
        metrics = self.results.setdefault("metrics", [])
        for metric in self.dependent_variables:
            if metric not in metrics:
                metrics.append(metric)

    @staticmethod
    def _get_summary_metric(summary, key, dep_var=None):
        """Resolve a metric from an evaluation summary, including fallbacks."""
        value = summary.get(key)
        if value is None and dep_var:
            value = summary.get(dep_var)
        if value is None and key == "f1_macro":
            report = summary.get("classification_report", {})
            macro_avg = report.get("macro avg", {})
            value = macro_avg.get("f1-score")
        return value

    @classmethod
    def resolve_checkpoint_path(cls, run_dir, checkpoint_preference):
        """Return the checkpoint path for a logging/evaluation preference."""
        if isinstance(checkpoint_preference, int):
            return Path(run_dir) / f"resnet18_epoch_{checkpoint_preference}.ckpt"

        checkpoint_name = cls._CHECKPOINT_FILES.get(checkpoint_preference)
        if checkpoint_name is None:
            raise ValueError(
                f"Invalid checkpoint preference '{checkpoint_preference}'. "
                "Expected one of: "
                f"{', '.join(sorted(cls._CHECKPOINT_FILES))}, or an epoch number."
            )
        return Path(run_dir) / checkpoint_name

    def expected_checkpoint_path(self, run_dir):
        """Return the configured checkpoint path for this run."""
        checkpoint_preference = self.config.get("logging", {}).get(
            "save_only", "lowest_val_loss"
        )
        return self.resolve_checkpoint_path(run_dir, checkpoint_preference)

    def summary_is_current_for_logging(self, summary_path, run_dir, split_name):
        """Return True when a summary can be reused for current logging."""
        summary_path = Path(summary_path)
        if not summary_path.exists():
            return False

        with open(summary_path, "r") as f:
            summary = json.load(f)

        expected_checkpoint = self.expected_checkpoint_path(run_dir)
        summary_checkpoint = summary.get("checkpoint")
        if summary_checkpoint is None:
            return False

        if (Path(summary_checkpoint).resolve(strict=False)
                != expected_checkpoint.resolve(strict=False)):
            return False

        for metric in (f"{split_name}_f1_score", f"{split_name}_f1_macro"):
            key = metric[len(f"{split_name}_"):]
            if self._get_summary_metric(summary, key, metric) is None:
                return False

        return True

    # ------------------------------------------------------------------ #
    #  Independent-variable helpers                                       #
    # ------------------------------------------------------------------ #

    def is_synthetic_split_test(self):
        """Return True when the independent variable is the synthetic count."""
        ind_var = (self.config.get("evaluation", {})
                   .get("independent_variable", ""))
        return ind_var == "evaluation.splits.synthetic_image_counts"

    def get_independent_variable_value(self, real_count, synthetic_count):
        """Derive the independent-variable value for a specific split."""
        logger.debug(self.config["evaluation"].keys())
        ind_var = (self.config.get("evaluation", {})
                   .get("independent_variable", ""))
        logger.debug("Getting independent variable value for "
              f"real_count={real_count}, synthetic_count={synthetic_count} "
              f"with independent_variable='{ind_var}'")
        if ind_var == "evaluation.splits.synthetic_image_counts":
            return synthetic_count
        if ind_var == "evaluation.splits.real_image_counts":
            return real_count
        value = self._resolve_config_value(self.config, ind_var)
        if value is None:
            logger.warning(f"[WARNING] Could not resolve independent variable "
                  f"'{ind_var}' from config — using "
                  f"'unknown_independent_variable'.")
            return "unknown_independent_variable"
        return value

    # ------------------------------------------------------------------ #
    #  Results initialisation & loading                                   #
    # ------------------------------------------------------------------ #

    def _init_results(self):
        """Build the initial results dict skeleton from the eval config."""
        evaluation = self.config.get("evaluation", {})
        eval_id = evaluation.get("evaluation_id", "unknown")
        ind_var = evaluation.get("independent_variable", "unknown")
        metrics = list(self.dependent_variables)

        x_label = self._pretty_label(ind_var)
        y_label = self._pretty_label(metrics[0]) if metrics else "Metric"

        return {
            "evaluation_id": eval_id,
            "independent_variable": ind_var,
            "independent_variable_value": None,
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

    def _load_or_init_results(self):
        """Load an existing results.json or create a fresh skeleton."""
        if self.results_path.exists():
            with open(self.results_path, "r") as f:
                logger.info(f"Loaded existing results from "
                      f"{self.results_path}")
                return json.load(f)
        return self._init_results()

    def _prepopulate_splits(self):
        """Create split entries for every configured split so independent
        variable values are recorded before any training begins."""
        splits_cfg = (self.config.get("evaluation", {})
                      .get("splits", {}))
        real_counts = splits_cfg.get("real_image_counts", [])
        synthetic_counts = splits_cfg.get("synthetic_image_counts", [])

        # Store independent variable value at top level if not yet set
        if real_counts and synthetic_counts:
            ind_val_first = self.get_independent_variable_value(
                real_counts[0], synthetic_counts[0])
            if self.results.get("independent_variable_value") is None:
                self.results["independent_variable_value"] = ind_val_first

        for rc, sc in zip(real_counts, synthetic_counts):
            ind_val = self.get_independent_variable_value(rc, sc)
            self.find_or_create_split_entry(rc, sc, ind_val)

        self.save()

    # ------------------------------------------------------------------ #
    #  Split entries                                                      #
    # ------------------------------------------------------------------ #

    def find_or_create_split_entry(self, real_count, synthetic_count,
                                   independent_value=None):
        """Return the matching split entry, creating one if needed."""
        for entry in self.results["splits"]:
            if (entry.get("real_image_count") == real_count
                    and entry.get("synthetic_image_count") == synthetic_count):
                return entry

        entry = {
            "split_name": f"{real_count}-real_{synthetic_count}-synthetic",
            "real_image_count": real_count,
            "synthetic_image_count": synthetic_count,
            "runs": [],
        }
        self.results["splits"].append(entry)
        return entry

    # ------------------------------------------------------------------ #
    #  Metric extraction                                                  #
    # ------------------------------------------------------------------ #

    def extract_run_metrics(self, val_summary_path, test_summary_path=None,
                            model_instance=None, run_dir=None):
        """Read dependent-variable metrics from val/test evaluation summaries.

        Metric routing is config-driven:
            - val_*  -> validation summary
            - test_* -> test summary
            - unprefixed metrics -> validation summary (legacy behavior)
        """
        val_summary = {}
        test_summary = {}

        if val_summary_path and Path(val_summary_path).exists():
            with open(val_summary_path, "r") as f:
                val_summary = json.load(f)

        if test_summary_path and Path(test_summary_path).exists():
            with open(test_summary_path, "r") as f:
                test_summary = json.load(f)

        run_metrics = {}
        for dep_var in self.dependent_variables:
            if dep_var == "val_loss":
                if (model_instance is not None
                        and hasattr(model_instance, "lowest_val_loss")):
                    run_metrics[dep_var] = round(
                        float(model_instance.lowest_val_loss), 6)
                elif run_dir is not None:
                    ckpt_path = (Path(run_dir)
                                 / "resnet18_lowest_val_loss.ckpt")
                    if ckpt_path.exists():
                        ckpt = torch.load(str(ckpt_path),
                                          map_location="cpu",
                                          weights_only=False)
                        val_losses = ckpt.get("val_loss", [])
                        if val_losses:
                            run_metrics[dep_var] = round(
                                float(min(val_losses)), 6)
            elif dep_var.startswith("val_"):
                key = dep_var[len("val_"):]
                value = self._get_summary_metric(val_summary, key, dep_var)
                if value is None:
                    logger.warning(f"[WARNING] Metric '{dep_var}' was requested "
                                  f"but not found in val summary")
                    continue
                run_metrics[dep_var] = round(float(value), 6)

            elif dep_var.startswith("test_"):
                key = dep_var[len("test_"):]
                value = self._get_summary_metric(test_summary, key, dep_var)
                if value is None:
                    logger.warning(f"[WARNING] Metric '{dep_var}' was requested "
                                  f"but not found in test summary")
                    continue
                run_metrics[dep_var] = round(float(value), 6)

            elif dep_var in self._METRIC_MAP:
                key = self._METRIC_MAP[dep_var]
                if key in val_summary:
                    run_metrics[dep_var] = round(
                        float(val_summary[key]), 6)
            else:
                if dep_var in val_summary:
                    try:
                        run_metrics[dep_var] = round(
                            float(val_summary[dep_var]), 6)
                    except (TypeError, ValueError):
                        logger.warning(f"[WARNING] Could not convert '{dep_var}' "
                              f"value to float")
                else:
                    logger.warning(f"[WARNING] Dependent variable '{dep_var}' "
                          f"not found in val summary")
        return run_metrics

    # ------------------------------------------------------------------ #
    #  High-level run logging                                             #
    # ------------------------------------------------------------------ #
 #Aris_4_class: 719364
 #deep ensembles: 653200
 # real dataset = 7 * 14330 = 100310
    def log_run(self, run_number, real_count, synthetic_count,
                val_summary_path, test_summary_path=None,
                model_instance=None, run_dir=None):
        """Extract metrics for a completed run and append them to results."""
        run_metrics = self.extract_run_metrics(
            val_summary_path,
            test_summary_path,
            model_instance,
            run_dir,
        )
        run_metrics["run_id"] = f"run{run_number}"
        
        # Extract seed from model instance or checkpoint
        seed = None
        if model_instance is not None and hasattr(model_instance, "seed"):
            seed = model_instance.seed
        elif run_dir is not None:
            # Try to read seed from checkpoint
            ckpt_path = self.expected_checkpoint_path(run_dir)
            if not ckpt_path.exists():
                ckpt_path = (Path(run_dir) / "resnet18_lowest_val_loss.ckpt")
            if ckpt_path.exists():
                try:
                    ckpt = torch.load(str(ckpt_path),
                                      map_location="cpu",
                                      weights_only=False)
                    seed = ckpt.get("seed")
                except Exception as e:
                    logger.warning(f"Could not extract seed from checkpoint: {e}")
        
        if seed is not None:
            run_metrics["seed"] = seed
            logger.info(f"Logged seed {seed} for run {run_number}")

        ind_value = self.get_independent_variable_value(
            real_count, synthetic_count)

        if self.results.get("independent_variable_value") is None:
            self.results["independent_variable_value"] = ind_value

        split_entry = self.find_or_create_split_entry(
            real_count, synthetic_count, ind_value)
        split_entry["runs"].append(run_metrics)
        self.save()

    # ------------------------------------------------------------------ #
    #  Export                                                            #
    # ------------------------------------------------------------------ #

    def save(self):
        """Atomically write the results dict to a JSON file."""
        with open(self.results_path, "w") as f:
            json.dump(self.results, f, indent=4)
        # tmp_path.rename(self.results_path)
        logger.info(f"Results saved to {self.results_path}")

    def export_csv(self, csv_path):
        """Export the results to a flat CSV (one row per run)."""
        dep_vars = self.results.get("metrics", [])
        ind_var = self.results.get("independent_variable", "unknown")
        eval_id = self.results.get("evaluation_id", "unknown")

        rows = []
        for entry in self.results.get("splits", []):
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
            logger.warning(f"[WARNING] No data to export to CSV at path {csv_path}")
            return

        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"CSV exported to {csv_path}")


if __name__ == "__main__":
    # Logging
    # logging.basicConfig(level=logging.INFO,
    #                     format="[%(asctime)s] %(levelname)s: %(message)s",
    #                     datefmt="%Y-%m-%d %H:%M:%S")
    config_path = ("/home/ap/cloud/Master/aris_master/queue/scheduled/resolution_128.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    logger = ResultsLogger(config)
    logger.info(f"Results path: {logger.results_path}")
    logger.info(f"Dependent variables: {logger.dependent_variables}")
    logger.info(f"Splits: {len(logger.results.get('splits', []))}")
