"""
Data Preprocessing Script for NYC 311 Service Requests

This script loads the downloaded data and creates proper train/test/validation splits
based on the available date range.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os



# Remove extreme outliers from the dataset using multiple methods.
def remove_extreme_outliers(df, target_col='resolution_time_hours', method='iqr', verbose=True):

    initial_count = len(df)
    df_clean = df.copy()
    
    if verbose:
        print(f"\n  Original data statistics for {target_col}:")
        print(f"    Count: {initial_count:,}")
        print(f"    Mean: {df[target_col].mean():.2f} hours")
        print(f"    Median: {df[target_col].median():.2f} hours")
        print(f"    Std Dev: {df[target_col].std():.2f} hours")
        print(f"    Min: {df[target_col].min():.2f} hours")
        print(f"    Max: {df[target_col].max():.2f} hours")
        print(f"    25th percentile: {df[target_col].quantile(0.25):.2f} hours")
        print(f"    75th percentile: {df[target_col].quantile(0.75):.2f} hours")
    
    if method == 'iqr':
        # IQR (Interquartile Range) method - most commonly used
        Q1 = df[target_col].quantile(0.25)
        Q3 = df[target_col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define bounds (3 * IQR is very conservative, removes extreme outliers)
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        if verbose:
            print(f"\n  IQR Method (3 × IQR):")
            print(f"    Q1 (25th percentile): {Q1:.2f} hours")
            print(f"    Q3 (75th percentile): {Q3:.2f} hours")
            print(f"    IQR: {IQR:.2f} hours")
            print(f"    Lower bound: {lower_bound:.2f} hours")
            print(f"    Upper bound: {upper_bound:.2f} hours")
        
        outlier_mask = (df_clean[target_col] < lower_bound) | (df_clean[target_col] > upper_bound)
        
    elif method == 'zscore':
        # Z-score method - remove values with |z-score| > 3 (99.7% confidence)
        mean = df[target_col].mean()
        std = df[target_col].std()
        
        if verbose:
            print(f"\n  Z-Score Method (|Z| > 3):")
            print(f"    Mean: {mean:.2f} hours")
            print(f"    Std Dev: {std:.2f} hours")
        
        z_scores = np.abs((df_clean[target_col] - mean) / std)
        outlier_mask = z_scores > 3
        
    elif method == 'percentile':
        # Percentile method - remove top and bottom 1%
        lower_bound = df[target_col].quantile(0.01)
        upper_bound = df[target_col].quantile(0.99)
        
        if verbose:
            print(f"\n  Percentile Method (1st-99th percentiles):")
            print(f"    Lower bound (1st percentile): {lower_bound:.2f} hours")
            print(f"    Upper bound (99th percentile): {upper_bound:.2f} hours")
        
        outlier_mask = (df_clean[target_col] < lower_bound) | (df_clean[target_col] > upper_bound)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Count and display outliers
    outlier_count = outlier_mask.sum()
    outlier_percentage = (outlier_count / initial_count) * 100
    
    if verbose:
        print(f"\n  Outliers detected: {outlier_count:,} records ({outlier_percentage:.2f}%)")
        if outlier_count > 0:
            outlier_data = df[outlier_mask][target_col]
            print(f"    Outlier range: {outlier_data.min():.2f} to {outlier_data.max():.2f} hours")
            print(f"    Mean outlier value: {outlier_data.mean():.2f} hours")
    
    # Remove outliers
    df_clean = df_clean[~outlier_mask].reset_index(drop=True)
    
    if verbose:
        print(f"\n  After outlier removal:")
        print(f"    Records remaining: {len(df_clean):,}")
        print(f"    Records removed: {outlier_count:,}")
        print(f"    Mean: {df_clean[target_col].mean():.2f} hours")
        print(f"    Median: {df_clean[target_col].median():.2f} hours")
        print(f"    Std Dev: {df_clean[target_col].std():.2f} hours")
        print(f"    Min: {df_clean[target_col].min():.2f} hours")
        print(f"    Max: {df_clean[target_col].max():.2f} hours")
    
    outlier_stats = {
        'method': method,
        'initial_count': initial_count,
        'final_count': len(df_clean),
        'outliers_removed': outlier_count,
        'removal_percentage': outlier_percentage,
        'original_mean': df[target_col].mean(),
        'original_median': df[target_col].median(),
        'original_std': df[target_col].std(),
        'original_min': df[target_col].min(),
        'original_max': df[target_col].max(),
        'cleaned_mean': df_clean[target_col].mean(),
        'cleaned_median': df_clean[target_col].median(),
        'cleaned_std': df_clean[target_col].std(),
        'cleaned_min': df_clean[target_col].min(),
        'cleaned_max': df_clean[target_col].max()
    }
    
    return df_clean, outlier_stats


def load_and_reprocess_data():

    print("Loading downloaded data...")

    # Load the three parts
    train = pd.read_csv("data/train_data.csv")
    test  = pd.read_csv("data/test_data.csv")
    val   = pd.read_csv("data/val_data.csv")
    val_df = pd.concat([train, test, val], ignore_index=True)
    val_df = val_df.drop_duplicates()
    print("Full dataset shape:", val_df.shape)

# Save combined full dataset for preprocessing
    val_df.to_csv("data/full_data.csv", index=False)
    print("Saved combined full dataset to data/full_data.csv")

    # Convert date columns
    val_df['created_date'] = pd.to_datetime(val_df['created_date'])
    val_df['closed_date'] = pd.to_datetime(val_df['closed_date'])
    
    # Sort by created_date
    val_df = val_df.sort_values('created_date').reset_index(drop=True)
    
    print(f"Total records loaded: {len(val_df)}")
    print(f"Date range: {val_df['created_date'].min()} to {val_df['created_date'].max()}")
    
    val_df, outlier_stats= remove_extreme_outliers(val_df, target_col='resolution_time_hours', method='iqr', verbose=True)
    print(f"\nAfter outlier removal, total records: {len(val_df)}")

    # Create splits: 70% train, 15% test, 15% validation
    total_records = len(val_df)
    train_end_idx = int(0.70 * total_records)
    test_end_idx = int(0.85 * total_records)
    
    train_df = val_df.iloc[:train_end_idx].copy()
    test_df = val_df.iloc[train_end_idx:test_end_idx].copy()
    val_df_split = val_df.iloc[test_end_idx:].copy()
    
    print(f"\nData Splits:")
    print(f"  Train set: {len(train_df):,} records ({len(train_df)/total_records*100:.1f}%)")
    print(f"  Test set: {len(test_df):,} records ({len(test_df)/total_records*100:.1f}%)")
    print(f"  Validation set: {len(val_df_split):,} records ({len(val_df_split)/total_records*100:.1f}%)")
    
    print(f"\nResolution Time Statistics:")
    print(f"  Mean: {val_df['resolution_time_hours'].mean():.2f} hours")
    print(f"  Median: {val_df['resolution_time_hours'].median():.2f} hours")
    print(f"  Std Dev: {val_df['resolution_time_hours'].std():.2f} hours")
    print(f"  Min: {val_df['resolution_time_hours'].min():.2f} hours")
    print(f"  Max: {val_df['resolution_time_hours'].max():.2f} hours")
    
    return {
        'train': train_df,
        'test': test_df,
        'val': val_df_split,
        'full': val_df
    }


# Save the new splits
def save_new_splits(data_dict):

    print("\nSaving data splits:")
    
    # Save data
    print("\n  Saving data splits:")
    data_dict['train'].to_csv('data/train_data.csv', index=False)
    data_dict['test'].to_csv('data/test_data.csv', index=False)
    data_dict['val'].to_csv('data/val_data.csv', index=False)
    print(f" Train: {len(data_dict['train']):,} records")
    print(f" Test: {len(data_dict['test']):,} records")
    print(f" Validation: {len(data_dict['val']):,} records")
    
    # Update metadata
    metadata = {
        'download_date': datetime.now().isoformat(),
        'preprocessing_date': datetime.now().isoformat(),
        'train_records': int(len(data_dict['train'])),
        'test_records': int(len(data_dict['test'])),
        'val_records': int(len(data_dict['val'])),
        'total_records': int(len(data_dict['train']) + len(data_dict['test']) + len(data_dict['val'])),
        'date_range': {
            'start': str(data_dict['train']['created_date'].min()),
            'end': str(data_dict['val']['created_date'].max())
        },
        'resolution_time_stats': {
            'mean': float(round(data_dict['full']['resolution_time_hours'].mean(), 2)),
            'median': float(round(data_dict['full']['resolution_time_hours'].median(), 2)),
            'std': float(round(data_dict['full']['resolution_time_hours'].std(), 2)),
            'min': float(round(data_dict['full']['resolution_time_hours'].min(), 2)),
            'max': float(round(data_dict['full']['resolution_time_hours'].max(), 2))
        }
    }
    
    with open('data/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n All data splits saved successfully!")
    print(f" Metadata updated")



#Statistics summary
def data_summary(data_dict):

    print("\n" + "="*70)
    print("DATA SUMMARY")
    print("="*70)
    
    for name, df in [("Train", data_dict['train']), ("Test", data_dict['test']), ("Validation", data_dict['val'])]:
        print(f"\n{name} Set:")
        print(f"  Records: {len(df):,}")
        print(f"  Date range: {df['created_date'].min()} to {df['created_date'].max()}")
        print(f"  Resolution time stats (hours):")
        print(f"    Mean: {df['resolution_time_hours'].mean():.1f}")
        print(f"    Median: {df['resolution_time_hours'].median():.1f}")
        print(f"    Std: {df['resolution_time_hours'].std():.1f}")
        print(f"    Min: {df['resolution_time_hours'].min():.1f}")
        print(f"    Max: {df['resolution_time_hours'].max():.1f}")
    
    print("\n" + "="*70)



# Preprocessing main function
def main():

    try:
        # Load and reprocess data
        data_dict = load_and_reprocess_data()
        
        # Save splits
        save_new_splits(data_dict)
        
        # Print summary
        data_summary(data_dict)
        
        print("\n" + "="*70)
        print("DATA PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"\nTotal records processed: {len(data_dict['full']):,}")
        print(f"Train set: {len(data_dict['train']):,} records")
        print(f"Test set: {len(data_dict['test']):,} records")
        print(f"Validation set: {len(data_dict['val']):,} records")
        print(f"\nNext step: Run 04_feature_engineering.py")
        
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
    