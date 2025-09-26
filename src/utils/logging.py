"""
Logging utilities for the project.
"""

import os
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: str = "INFO",
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with both file and console handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
        format_string: Custom format string
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Default format
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file is provided)
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_experiment_logger(experiment_name: str, log_dir: str = "outputs/logs") -> logging.Logger:
    """
    Get a logger for a specific experiment.
    
    Args:
        experiment_name: Name of the experiment
        log_dir: Directory to store log files
        
    Returns:
        Configured logger for the experiment
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{experiment_name}_{timestamp}.log"
    log_path = os.path.join(log_dir, log_filename)
    
    return setup_logger(
        name=experiment_name,
        log_file=log_path,
        level="INFO"
    )


class TrainingLogger:
    """Logger specifically for training progress."""
    
    def __init__(self, experiment_name: str, log_dir: str = "outputs/logs"):
        self.logger = get_experiment_logger(experiment_name, log_dir)
        self.start_time = datetime.now()
        
    def log_epoch(self, epoch: int, train_loss: float, val_loss: float = None, 
                  train_metrics: dict = None, val_metrics: dict = None):
        """Log epoch results."""
        msg = f"Epoch {epoch}: Train Loss = {train_loss:.4f}"
        
        if val_loss is not None:
            msg += f", Val Loss = {val_loss:.4f}"
        
        if train_metrics:
            for key, value in train_metrics.items():
                msg += f", Train {key} = {value:.4f}"
        
        if val_metrics:
            for key, value in val_metrics.items():
                msg += f", Val {key} = {value:.4f}"
        
        self.logger.info(msg)
    
    def log_training_start(self, total_epochs: int, total_samples: int):
        """Log training start information."""
        self.logger.info(f"Starting training for {total_epochs} epochs")
        self.logger.info(f"Total training samples: {total_samples}")
        self.logger.info(f"Training started at: {self.start_time}")
    
    def log_training_complete(self):
        """Log training completion."""
        end_time = datetime.now()
        duration = end_time - self.start_time
        self.logger.info(f"Training completed at: {end_time}")
        self.logger.info(f"Total training time: {duration}")
    
    def log_model_save(self, model_path: str, metric_value: float = None):
        """Log model saving."""
        msg = f"Model saved to: {model_path}"
        if metric_value is not None:
            msg += f" (metric: {metric_value:.4f})"
        self.logger.info(msg)