"""
Test configuration loading and basic functionality.
"""

import os
import pytest
import tempfile
import yaml
from pathlib import Path

# Add src to path for testing
import sys
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from utils.config import Config, load_config, save_config


def test_default_config():
    """Test default configuration creation."""
    config = Config()
    
    assert config.data.batch_size == 8
    assert config.model.n_classes == 2
    assert config.training.epochs == 100
    assert config.evaluation.save_best_model is True


def test_config_save_load():
    """Test configuration save and load functionality."""
    config = Config()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        temp_path = f.name
        
    try:
        # Save config
        save_config(config, temp_path)
        assert os.path.exists(temp_path)
        
        # Load config
        loaded_config = load_config(temp_path)
        
        # Check that values match
        assert loaded_config.data.batch_size == config.data.batch_size
        assert loaded_config.model.n_classes == config.model.n_classes
        assert loaded_config.training.epochs == config.training.epochs
        
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_config_yaml_format():
    """Test that configuration can be loaded from YAML."""
    yaml_content = """
data:
  batch_size: 16
  image_size: [256, 256]
  
model:
  segmentation:
    name: "unet"
    n_classes: 3
    
training:
  epochs: 50
  learning_rate: 0.01
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        temp_path = f.name
    
    try:
        config = load_config(temp_path)
        assert config.data.batch_size == 16
        assert config.data.image_size == [256, 256]
        assert config.training.epochs == 50
        assert config.training.learning_rate == 0.01
        
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__])