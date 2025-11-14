@echo off
REM NYC 311 Service Request Resolution Time Prediction
REM Environment Setup Script for Windows
REM 
REM This script creates a Python virtual environment and installs all required dependencies.
REM Run this script before starting the project.

echo 🚀 NYC 311 Service Request Prediction - Environment Setup
echo ==========================================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python is not installed or not in PATH. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

echo ✅ Python detected

REM Create virtual environment
echo 📦 Creating virtual environment...
if exist ".venv" (
    echo ⚠️  Virtual environment already exists. Removing old environment...
    rmdir /s /q .venv
)

python -m venv .venv
echo ✅ Virtual environment created in .venv\

REM Activate virtual environment
echo 🔌 Activating virtual environment...
call .venv\Scripts\activate.bat

REM Upgrade pip
echo ⬆️  Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo 📋 Installing Python packages from requirements.txt...
pip install -r requirements.txt

REM Verify installation of key packages
echo 🔍 Verifying key package installations...
python -c "import pandas; import numpy; import sklearn; import matplotlib; import seaborn; print('✅ Core packages verified successfully')"

REM Try to import optional packages
echo 🔍 Checking optional packages...
python -c "try: import xgboost; print('✅ XGBoost available'); except ImportError: print('⚠️  XGBoost not available')"
python -c "try: import lightgbm; print('✅ LightGBM available'); except ImportError: print('⚠️  LightGBM not available')"
python -c "try: import tensorflow; print('✅ TensorFlow available'); except ImportError: print('⚠️  TensorFlow not available')"

echo.
echo 🎉 Environment setup completed successfully!
echo.
echo 📋 Next steps:
echo 1. Activate the environment: .venv\Scripts\activate.bat
echo 2. Run the project pipeline:
echo    python 01_data_download.py
echo    python 02_data_preprocessing.py
echo    python 03_exploratory_analysis.py
echo    python 04_feature_engineering.py
echo    python 05_machine_learning_models.py
echo    python 06_project_summary.py
echo.
echo 💡 To deactivate the environment later, run: deactivate
echo.
echo ✨ Happy modeling! ✨

pause