"""
Configuration management utilities.
"""

import yaml
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    """Data configuration."""
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    synthetic_data_dir: str = "data/synthetic"
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    image_size: list = field(default_factory=lambda: [512, 512])
    normalize: bool = True
    augmentation: bool = True
    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class ModelConfig:
    """Model configuration."""
    segmentation_name: str = "unet"
    n_channels: int = 3
    n_classes: int = 2
    pretrained: bool = False
    latent_dim: int = 100
    feature_map_size: int = 64


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    g_lr: float = 0.0002
    d_lr: float = 0.0002
    beta1: float = 0.5
    beta2: float = 0.999
    lambda_gp: float = 10
    loss_function: str = "cross_entropy"
    class_weights: list = field(default_factory=lambda: [1.0, 2.0])
    scheduler_type: str = "step"
    step_size: int = 30
    gamma: float = 0.1


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    metrics: list = field(default_factory=lambda: ["accuracy", "iou", "dice", "precision", "recall", "f1"])
    eval_frequency: int = 5
    save_best_model: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration."""
    log_dir: str = "outputs/logs"
    experiment_name: str = "waste_segmentation_experiment"
    log_frequency: int = 10
    use_wandb: bool = False
    wandb_project: str = "waste-segmentation"
    wandb_entity: Optional[str] = None


@dataclass
class OutputConfig:
    """Output configuration."""
    model_save_dir: str = "outputs/models"
    results_save_dir: str = "outputs/results"
    visualization_save_dir: str = "outputs/visualizations"
    checkpoint_frequency: int = 10


@dataclass
class HardwareConfig:
    """Hardware configuration."""
    device: str = "auto"
    mixed_precision: bool = True
    deterministic: bool = True
    benchmark: bool = True


@dataclass
class SyntheticConfig:
    """Synthetic data generation configuration."""
    num_samples: int = 1000
    generation_batch_size: int = 32
    fid_threshold: float = 50.0
    synthetic_augmentation: bool = True


@dataclass
class Config:
    """Main configuration class."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)


def load_config(config_path: str) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration object
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Create configuration objects
    config = Config()
    
    if 'data' in config_dict:
        config.data = DataConfig(**config_dict['data'])
    
    if 'model' in config_dict:
        model_dict = config_dict['model']
        # Flatten nested model config
        flat_model_dict = {}
        if 'segmentation' in model_dict:
            flat_model_dict.update({f"segmentation_{k}": v for k, v in model_dict['segmentation'].items()})
        if 'generator' in model_dict:
            flat_model_dict.update(model_dict['generator'])
        config.model = ModelConfig(**flat_model_dict)
    
    if 'training' in config_dict:
        training_dict = config_dict['training']
        # Flatten nested training config
        flat_training_dict = {}
        for key, value in training_dict.items():
            if isinstance(value, dict):
                flat_training_dict.update(value)
            else:
                flat_training_dict[key] = value
        # Handle scheduler separately
        if 'scheduler' in training_dict:
            flat_training_dict.update({f"scheduler_{k}": v for k, v in training_dict['scheduler'].items()})
        config.training = TrainingConfig(**flat_training_dict)
    
    if 'evaluation' in config_dict:
        config.evaluation = EvaluationConfig(**config_dict['evaluation'])
    
    if 'logging' in config_dict:
        config.logging = LoggingConfig(**config_dict['logging'])
    
    if 'output' in config_dict:
        config.output = OutputConfig(**config_dict['output'])
    
    if 'hardware' in config_dict:
        config.hardware = HardwareConfig(**config_dict['hardware'])
    
    if 'synthetic' in config_dict:
        config.synthetic = SyntheticConfig(**config_dict['synthetic'])
    
    return config


def save_config(config: Config, save_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration object
        save_path: Path to save the configuration file
    """
    # Convert dataclasses to dictionaries
    config_dict = {
        'data': config.data.__dict__,
        'model': config.model.__dict__,
        'training': config.training.__dict__,
        'evaluation': config.evaluation.__dict__,
        'logging': config.logging.__dict__,
        'output': config.output.__dict__,
        'hardware': config.hardware.__dict__,
        'synthetic': config.synthetic.__dict__,
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)


def create_directories(config: Config) -> None:
    """
    Create necessary directories based on configuration.
    
    Args:
        config: Configuration object
    """
    directories = [
        config.data.raw_data_dir,
        config.data.processed_data_dir,
        config.data.synthetic_data_dir,
        config.logging.log_dir,
        config.output.model_save_dir,
        config.output.results_save_dir,
        config.output.visualization_save_dir,
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")