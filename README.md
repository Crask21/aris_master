# Synthetic Data Generation for Waste Segmentation

A master's thesis project focused on implementing state-of-the-art image generation algorithms to enhance waste segmentation performance on conveyor belt systems through synthetic data augmentation.

## 🎯 Project Overview

This project aims to improve the performance of waste segmentation algorithms by generating synthetic training data using Generative Adversarial Networks (GANs). The system processes images of conveyor belts with scattered waste pieces and generates realistic synthetic samples to augment the training dataset.

## 🏗️ Project Structure

```
aris_master/
├── src/                          # Source code
│   ├── data/                     # Data loading and preprocessing
│   ├── models/                   # Model architectures (GAN, U-Net, etc.)
│   ├── training/                 # Training scripts
│   ├── evaluation/               # Evaluation metrics and utilities
│   └── utils/                    # Utility functions
├── config/                       # Configuration files
├── data/                         # Data directories
│   ├── raw/                      # Raw dataset
│   ├── processed/                # Preprocessed data
│   └── synthetic/                # Generated synthetic data
├── experiments/                  # Experiment configurations and results
├── notebooks/                    # Jupyter notebooks for analysis
├── scripts/                      # Utility scripts
├── outputs/                      # Model outputs and results
│   ├── models/                   # Trained models
│   ├── results/                  # Evaluation results
│   └── visualizations/           # Generated plots and images
└── tests/                        # Unit tests
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- CUDA-compatible GPU (recommended)
- UV package manager (already setup in your project)

### Installation

1. **Install dependencies using UV:**
   ```bash
   uv sync
   ```

2. **Activate the virtual environment:**
   ```bash
   # On Windows
   .venv\Scripts\activate
   
   # On Linux/Mac
   source .venv/bin/activate
   ```

### Basic Usage

1. **Set up the project structure:**
   ```bash
   python main.py setup --config config/config.yaml
   ```

2. **Prepare your data:**
   ```bash
   python main.py prepare-data --config config/config.yaml --source path/to/your/raw/data
   ```

3. **Train a segmentation model:**
   ```bash
   python main.py train-segmentation --config config/config.yaml
   ```

4. **Train a GAN for synthetic data generation:**
   ```bash
   python main.py train-gan --config config/config.yaml
   ```

5. **Generate synthetic data:**
   ```bash
   python main.py generate --config config/config.yaml --model outputs/models/final_generator.pth
   ```

6. **Evaluate the model:**
   ```bash
   python main.py evaluate --config config/config.yaml --model outputs/models/best_model.pth
   ```

## 📊 Dataset Format

The expected dataset structure:
```
data/raw/
├── images/           # RGB images of conveyor belts
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
└── masks/            # Segmentation masks
    ├── image001.png  # Binary masks (0: background, 1: waste)
    ├── image002.png
    └── ...
```

## 🛠️ Configuration

The project uses YAML configuration files. The main configuration file is located at `config/config.yaml`. Key sections include:

- **Data Configuration**: Dataset paths, preprocessing parameters
- **Model Configuration**: Architecture settings for segmentation and GAN models
- **Training Configuration**: Learning rates, batch sizes, epochs
- **Evaluation Configuration**: Metrics and validation settings

## 🔬 Models

### Segmentation Models
- **U-Net**: Classic encoder-decoder architecture for semantic segmentation
- **DeepLabV3+**: Advanced segmentation model with atrous convolutions

### Generative Models
- **DCGAN**: Deep Convolutional GAN for synthetic image generation
- **Custom WasteGAN**: Specialized architecture for waste image generation

## 📈 Evaluation Metrics

The project includes comprehensive evaluation metrics:
- Pixel Accuracy
- Intersection over Union (IoU)
- Dice Coefficient
- Precision, Recall, F1-Score
- Per-class metrics

## 🧪 Experiments

Use the `experiments/` directory to store different experimental configurations and results. Each experiment can have its own configuration file and output directory.

## 📓 Notebooks

The `notebooks/` directory contains Jupyter notebooks for:
- Data exploration and visualization
- Model analysis and debugging
- Results visualization and interpretation

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

## 📝 Logging and Monitoring

The project supports multiple logging options:
- Local file logging
- Weights & Biases (wandb) integration for experiment tracking

## 🤝 Contributing

1. Follow the existing code structure and naming conventions
2. Add tests for new functionality
3. Update documentation as needed
4. Use Black for code formatting: `black src/`
