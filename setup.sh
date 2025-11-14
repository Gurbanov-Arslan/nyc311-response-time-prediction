#!/bin/bash

# NYC 311 Service Request Resolution Time Prediction
# Environment Setup Script
# 
# This script creates a Python virtual environment and installs all required dependencies.
# Run this script before starting the project.

set -e  # Exit on any error

echo "🚀 NYC 311 Service Request Prediction - Environment Setup"
echo "=========================================================="

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Error: Python $REQUIRED_VERSION or higher is required. Found: $PYTHON_VERSION"
    exit 1
fi

echo "✅ Python $PYTHON_VERSION detected"

# Create virtual environment
echo "📦 Creating virtual environment..."
if [ -d ".venv" ]; then
    echo "⚠️  Virtual environment already exists. Removing old environment..."
    rm -rf .venv
fi

python3 -m venv .venv
echo "✅ Virtual environment created in .venv/"

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📋 Installing Python packages from requirements.txt..."
pip install -r requirements.txt

# Verify installation of key packages
echo "🔍 Verifying key package installations..."
python3 -c "
import pandas
import numpy
import sklearn
import matplotlib
import seaborn
print('✅ Core packages verified successfully')
"

# Try to import optional packages
echo "🔍 Checking optional packages..."
python3 -c "
try:
    import xgboost
    print('✅ XGBoost available')
except ImportError:
    print('⚠️  XGBoost not available - installing OpenMP if needed...')
    
try:
    import lightgbm
    print('✅ LightGBM available')
except ImportError:
    print('⚠️  LightGBM not available')

try:
    import tensorflow
    print('✅ TensorFlow available')
except ImportError:
    print('⚠️  TensorFlow not available')
"

echo ""
echo "🎉 Environment setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Activate the environment: source .venv/bin/activate"
echo "2. Run the project pipeline:"
echo "   python 01_data_download.py"
echo "   python 02_data_preprocessing.py"
echo "   python 03_exploratory_analysis.py"
echo "   python 04_feature_engineering.py"
echo "   python 05_machine_learning_models.py"
echo "   python 06_project_summary.py"
echo ""
echo "💡 To deactivate the environment later, run: deactivate"
echo ""

# Check if we're on macOS and XGBoost might need OpenMP
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "🍎 macOS detected. If XGBoost fails, install OpenMP with:"
    echo "   brew install libomp"
    echo ""
fi

echo "✨ Happy modeling! ✨"