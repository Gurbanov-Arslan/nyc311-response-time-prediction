#!/usr/bin/env python3
"""
NYC 311 Service Request Resolution Time Prediction
Complete Pipeline Runner
This script runs the entire machine learning pipeline from data download to final results.
"""

import subprocess
import sys
import os
from datetime import datetime


# Function to run a command and handle errors 
def run_command(command, description):
    
    print(f"\n{'='*60}")
    print(f" {description}")
    print(f"{'='*60}")
    print(f"Running: python {command}")
    
    try:
        result = subprocess.run([sys.executable, command], 
                              capture_output=False, 
                              text=True, 
                              check=True)
        print(f" {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error in {description}: {e}")
        return False
    except FileNotFoundError:
        print(f"File not found: {command}")
        return False



 #Run the complete pipeline
def main():
    start_time = datetime.now()
    
    print("NYC 311 Service Request Resolution Time Prediction")
    print("Complete Machine Learning Pipeline")
    print("=" * 60)
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if we're in the right directory
    required_files = [
        '01_data_download.py',
        '02_data_preprocessing.py',
        '03_exploratory_analysis.py',
        '04_feature_engineering.py',
        '05_machine_learning_models.py',
        '06_project_summary.py',
        '07_model_performance_by_category.py',
        '08_model_performance_by_time_bins.py'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f" Missing required files: {missing_files}")
        print("Please run this script from the project root directory.")
        return False
    
    # Pipeline steps
    steps = [
        ('01_data_download.py', 'Data Download from NYC Open Data API'),
        ('02_data_preprocessing.py', 'Data Preprocessing and Splitting'),
        ('03_exploratory_analysis.py', 'Exploratory Data Analysis'),
        ('04_feature_engineering.py', 'Feature Engineering'),
        ('05_machine_learning_models.py', 'Machine Learning Model Training'),
        ('06_project_summary.py', 'Project Summary Generation'),
        ('07_model_performance_by_category.py', 'Model Performance by Category Analysis'),
        ('08_model_performance_by_time_bins.py', 'Model Performance by Resolution Time Bins Analysis')
    ]
    
    # Run each step
    success_count = 0
    for script, description in steps:
        if run_command(script, description):
            success_count += 1
        else:
            print(f"\n Pipeline failed at step: {description}")
            print("Please check the error messages above and fix any issues.")
            return False
    
    # Final summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n{'='*60}")
    print(f"\n PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    print(f" All 8 steps completed successfully")
    print(f"  Total duration: {duration}")
    print(f" Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\n Output directories created:")
    print(f"  data/ - Raw and preprocessed datasets")
    print(f"  features/ - Feature-engineered datasets")
    print(f"  models/ - Trained models and performance metrics")
    print(f"  plots/ - Visualizations and analysis charts")
    

    print(f"\n Next Steps:")
    print(f"   Review plots/ directory for visualizations")
    print(f"   Check models/ directory for detailed results")
    print(f"   Consider model deployment for production use")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)