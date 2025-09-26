"""
Basic tests for data loading functionality.
"""

import pytest
import numpy as np
import torch
from PIL import Image
import tempfile
import os

# Add src to path for testing
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from data.preprocessing import preprocess_image, preprocess_mask


def test_preprocess_image():
    """Test image preprocessing functionality."""
    # Create a dummy image
    dummy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # Test preprocessing
    processed = preprocess_image(dummy_image, target_size=(64, 64), normalize=True)
    
    assert processed.shape == (64, 64, 3)
    assert processed.dtype == np.float32
    assert processed.min() >= 0.0
    assert processed.max() <= 1.0


def test_preprocess_mask():
    """Test mask preprocessing functionality."""
    # Create a dummy mask
    dummy_mask = np.random.randint(0, 2, (100, 100), dtype=np.uint8)
    
    # Test preprocessing
    processed = preprocess_mask(dummy_mask, target_size=(64, 64), num_classes=2)
    
    assert processed.shape == (64, 64)
    assert processed.dtype == np.uint8
    assert processed.min() >= 0
    assert processed.max() < 2


def test_image_normalization():
    """Test image normalization."""
    # Create image with known values
    test_image = np.full((32, 32, 3), 128, dtype=np.uint8)  # All pixels = 128
    
    # Preprocess with normalization
    processed = preprocess_image(test_image, normalize=True)
    
    # Should be approximately 0.5 (128/255)
    expected_value = 128.0 / 255.0
    assert np.allclose(processed, expected_value, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])