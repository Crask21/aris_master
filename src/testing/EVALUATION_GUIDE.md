# ResNet18 Evaluation Guide

## Overview

The `evaluate_resnet18.py` script provides comprehensive model evaluation capabilities for ResNet18 classifiers. It automatically uses test data if available, otherwise falls back to validation data.

## Features

✓ Automatic test/validation split detection  
✓ Comprehensive metrics (accuracy, precision, recall, F1)  
✓ Predictions CSV export with probabilities  
✓ Confusion matrix visualization (raw and normalized)  
✓ Classification report with per-class metrics  
✓ Both CLI and programmatic usage  

## Usage

### 1. Command Line Evaluation

Evaluate a trained model directly:

```bash
python src/testing/evaluate_resnet18.py --config path/to/config.json
```

Specify a custom checkpoint:

```bash
python src/testing/evaluate_resnet18.py \
    --config path/to/config.json \
    --checkpoint path/to/checkpoint.ckpt
```

Specify custom output directory:

```bash
python src/testing/evaluate_resnet18.py \
    --config path/to/config.json \
    --output-dir path/to/output
```

### 2. Evaluate After Training

Add the `--evaluate` flag to automatically evaluate after training completes:

```bash
python src/testing/train_resnet18.py \
    --config path/to/config.json \
    --evaluate
```

This will:
- Complete training normally
- Load the best validation accuracy checkpoint
- Run comprehensive evaluation
- Save all results to the output directory

### 3. Programmatic Usage

Import and use in your Python code:

```python
from src.testing.evaluate_resnet18 import evaluate_resnet18

# Basic usage (uses config settings)
results = evaluate_resnet18(config_path='path/to/config.json')

# With custom checkpoint
results = evaluate_resnet18(
    config_path='path/to/config.json',
    checkpoint_path='path/to/model.ckpt'
)

# With custom output directory
results = evaluate_resnet18(
    config_path='path/to/config.json',
    checkpoint_path='path/to/model.ckpt',
    output_dir='path/to/evaluation_results'
)

# Access results
print(f"Accuracy: {results['accuracy']*100:.2f}%")
print(f"F1 Score: {results['f1']:.4f}")
print(f"Predictions: {results['predictions']}")
```

## Output Files

The evaluation script generates several output files in the evaluation directory:

### 1. predictions_[split].csv
Contains per-sample predictions with:
- True label
- Predicted label
- Correctness flag
- Probability for each class
- Filepath (if available)

Example:
```csv
true_label,predicted_label,correct,prob_wood,prob_plastic,filepath
wood,wood,True,0.95,0.05,/path/to/img1.png
plastic,plastic,True,0.12,0.88,/path/to/img2.png
```

### 2. confusion_matrix_[split].png
Visual confusion matrix showing:
- Raw counts of predictions
- True labels vs predicted labels
- Helpful for identifying misclassification patterns

### 3. confusion_matrix_[split]_normalized.png
Normalized confusion matrix showing:
- Percentage of predictions
- Better for imbalanced datasets

### 4. classification_report_[split].txt
Detailed text report with:
- Per-class precision, recall, F1-score
- Support (number of samples per class)
- Overall weighted averages
- Summary metrics

Example:
```
Classification Report
================================================================================

              precision    recall  f1-score   support

        wood     0.9250    0.9487    0.9367        39
     plastic     0.9459    0.9211    0.9333        38

    accuracy                         0.9351        77
   macro avg     0.9355    0.9349    0.9350        77
weighted avg     0.9354    0.9351    0.9350        77
```

### 5. evaluation_summary_[split].json
JSON summary with key metrics:
```json
{
    "split": "test",
    "checkpoint": "path/to/checkpoint.ckpt",
    "num_samples": 275,
    "num_classes": 3,
    "class_names": ["wood", "plastic", "metal"],
    "accuracy": 0.9351,
    "precision": 0.9354,
    "recall": 0.9351,
    "f1_score": 0.9350
}
```

## Test vs Validation Split

The script automatically detects which split to use:

1. **Test split preferred**: If `data_dir/test/` exists with images, uses test split
2. **Validation fallback**: If no test split found, uses validation split
3. **Clear indication**: Prints which split is being used during evaluation

## Requirements

The evaluation function requires:
- Trained ResNet18 checkpoint
- Config file with data paths and classes
- Test or validation dataset
- PyTorch, torchvision, pandas, seaborn, scikit-learn

## Integration with Training

The evaluation is seamlessly integrated with training:

```python
# In train_resnet18.py
if args.evaluate and evaluate_resnet18 is not None:
    # Uses best validation accuracy checkpoint
    best_checkpoint = os.path.join(output_dir, "resnet18_best_val_acc.ckpt")
    evaluate_resnet18(config_path=config_path, checkpoint_path=best_checkpoint)
```

## Return Value

When called programmatically, returns a dictionary with:

```python
{
    "predictions": np.array,      # Predicted class indices
    "labels": np.array,           # True class indices  
    "probabilities": np.array,    # Class probabilities (N x num_classes)
    "filepaths": list,            # Image filepaths
    "accuracy": float,            # Overall accuracy
    "precision": float,           # Weighted precision
    "recall": float,              # Weighted recall
    "f1": float,                  # Weighted F1 score
    "class_names": list,          # Class names
    "split_name": str,            # "test" or "val"
    "checkpoint_path": str,       # Path to checkpoint used
    "config_path": str            # Path to config used
}
```

## Example Workflow

Complete workflow from training to evaluation:

```bash
# 1. Train the model with automatic evaluation
python src/testing/train_resnet18.py \
    --config config.json \
    --evaluate

# Or train first, then evaluate separately
python src/testing/train_resnet18.py --config config.json

# 2. Evaluate on test set
python src/testing/evaluate_resnet18.py \
    --config config.json \
    --checkpoint training/resnet18/resnet18_best_val_acc.ckpt

# 3. Check results
ls training/evaluation/
# predictions_test.csv
# confusion_matrix_test.png
# confusion_matrix_test_normalized.png
# classification_report_test.txt
# evaluation_summary_test.json
```

## Troubleshooting

**No checkpoint found**: Ensure the config file has `logging.checkpoint_dir` set, or pass `--checkpoint` explicitly.

**No test data**: Script will automatically use validation data. Check console output for which split is being used.

**Import errors**: Ensure you're running from the project root and all dependencies are installed.

**CUDA out of memory**: Reduce batch size in config file or use CPU by setting `CUDA_VISIBLE_DEVICES=""`.
