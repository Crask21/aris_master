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
        "val_accuracy": "accuracy",
        "f1_score": "f1_score",
        "precision": "precision",
        "recall": "recall",
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

        # Load or initialise results
        self.results = self._load_or_init_results()

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

    def extract_run_metrics(self, eval_summary_path, model_instance=None,
                            run_dir=None):
        """Read dependent-variable metrics from the evaluation summary (and
        optionally from the training model / checkpoint for val_loss)."""
        eval_summary = {}
        if Path(eval_summary_path).exists():
            with open(eval_summary_path, "r") as f:
                eval_summary = json.load(f)

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
            elif dep_var in self._METRIC_MAP:
                key = self._METRIC_MAP[dep_var]
                if key in eval_summary:
                    run_metrics[dep_var] = round(
                        float(eval_summary[key]), 6)
            else:
                if dep_var in eval_summary:
                    try:
                        run_metrics[dep_var] = round(
                            float(eval_summary[dep_var]), 6)
                    except (TypeError, ValueError):
                        logger.warning(f"[WARNING] Could not convert '{dep_var}' "
                              f"value to float")
                else:
                    logger.warning(f"[WARNING] Dependent variable '{dep_var}' "
                          f"not found in evaluation summary")
        return run_metrics

    # ------------------------------------------------------------------ #
    #  High-level run logging                                             #
    # ------------------------------------------------------------------ #

    def log_run(self, run_number, real_count, synthetic_count,
                eval_summary_path, model_instance=None, run_dir=None):
        """Extract metrics for a completed run and append them to results."""
        run_metrics = self.extract_run_metrics(
            eval_summary_path, model_instance, run_dir)
        run_metrics["run_id"] = f"run{run_number}"

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