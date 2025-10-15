#!/bin/bash
set -e

echo "DA-Fusion Installation Script"
echo "================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    print_error "UV package manager not found. Please install it first:"
    echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

print_status "UV package manager found"

# Check if we're in the da-fusion directory
if [ ! -f "setup.py" ] || [ ! -d "semantic_aug" ]; then
    print_error "This script must be run from the da-fusion directory"
    echo "Please clone the repository and navigate to the da-fusion directory."
    echo "git clone https://github.com/brandontrabucco/da-fusion.git"
    echo "Afterwards run: cd da-fusion && bash ../da-fusion_install.sh"
    exit 1
fi

print_status "Found da-fusion project directory"

# Create and activate virtual environment
print_status "Creating virtual environment..."
uv init --name da-fusion
uv venv

# Sync remaining dependencies
print_status "Syncing with uv..."
uv sync

# Activate virtual environment and install package in editable mode
print_status "Installing semantic-aug package in editable mode..."

# Source the virtual environment
source .venv/bin/activate

# Install pip if not available
if ! command -v pip &> /dev/null; then
    print_status "Installing pip in virtual environment..."
    python -m ensurepip --upgrade
fi

# Temporarily move pyproject.toml to avoid conflicts
print_warning "Temporarily moving pyproject.toml to avoid setup conflicts..."
mv pyproject.toml pyproject.toml.backup

# Install package in editable mode using setup.py
print_status "Installing package in editable mode..."
python -m pip install -e .

# Restore pyproject.toml
print_status "Restoring pyproject.toml..."
mv pyproject.toml.backup pyproject.toml

# Verify installation
print_status "Verifying installation..."

# Test basic import
python -c "import semantic_aug; print('✅ semantic_aug imports successfully')" || {
    print_error "Basic import failed!"
    exit 1
}

# Test augmentation imports  
python -c "from semantic_aug.augmentations.textual_inversion import TextualInversion; print('✅ TextualInversion imports successfully')" || {
    print_error "TextualInversion import failed!"
    exit 1
}

# Test transformers imports
python -c "from transformers import AutoImageProcessor, DeiTModel; print('✅ Transformers imports successfully')" || {
    print_error "Transformers import failed!"
    exit 1
}

# Test script functionality
python train_classifier.py --help > /dev/null || {
    print_error "train_classifier.py --help failed!"
    exit 1
}

# Test train_classifier with --help
python train_classifier.py --help > /dev/null || {
    print_error "train_classifier.py --help failed!"
    exit 1
}

echo ""
echo "🎉 Installation completed successfully!"
echo ""
echo "Working package versions installed:"
echo "  - torch==1.12.1+cu116 & torchvision==0.13.1+cu116"
echo "  - diffusers==0.14.0 (compatible with PyTorch 1.12.1)"
echo "  - transformers==4.25.1 (compatible with PyTorch 1.12.1)"
echo "  - accelerate==0.16.0 & huggingface-hub==0.13.4"
echo ""
echo "To use the environment:"
echo "  source .venv/bin/activate"
echo "  python train_classifier.py --help"
echo ""
echo "Example usage:"
echo "  python train_classifier.py --dataset pascal --num-trials 1 --examples-per-class 4"
