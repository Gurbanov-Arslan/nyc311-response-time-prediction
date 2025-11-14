"""
NYC 311 Service Requests Data Download Script

This script downloads the NYC 311 service requests dataset from NYC Open Data,
filters for the last 2 years, and prepares it for analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sodapy import Socrata
import os
import json
import time


# Download NYC 311 data for the last 2 years
def download_311_data():

    print("Starting NYC 311 data download:")
    
    # NYC Open Data endpoint for 311 service requests
    # Increase timeout to 90 seconds to handle large requests
    client = Socrata("data.cityofnewyork.us", None, timeout=90)
    
    # Calculate date range - last 2 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # 2 years
    
    print(f"Downloading data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Define the query with date filter
    where_clause = f"created_date >= '{start_date.strftime('%Y-%m-%dT%H:%M:%S')}'"
    
    # Download data in chunks to handle large dataset
    data_chunk_size = 50000  
    downloaded_data_count = 0
    all_data = []
    max_retries = 3  # Retry failed requests
    
    while True:
        print(f"Downloading records {downloaded_data_count} to {downloaded_data_count + data_chunk_size}...")
        
        retry_count = 0
        while retry_count < max_retries:
            try:
                # Query the 311 dataset - order by created_date ASC to get older data first
                results = client.get("erm2-nwe9", 
                                   where=where_clause,
                                   limit=data_chunk_size,
                                   offset=downloaded_data_count,
                                   order="created_date ASC")
                
                if not results:
                    print("No more data to download.")
                    return all_data
                    
                all_data.extend(results)
                downloaded_data_count += data_chunk_size
                
                # Stop if we get less than the limit (last batch)
                if len(results) < data_chunk_size:
                    print(f"Reached end of data (last batch had {len(results)} records)")
                    return all_data
                    
                # Stop after a reaching desired number of data
                if len(all_data) >= 2000000:  # Download 2 million records
                    print(f"Reached target limit of 2000000 records")
                    return all_data
                
                # Success - break retry loop
                break
                
            except Exception as e:
                retry_count += 1
                print(f"Error downloading data (attempt {retry_count}/{max_retries}): {e}")
                
                if retry_count < max_retries:
                    wait_time = 5 * retry_count  # Wait 5, 10, 15 seconds
                    print(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    print(f"Failed after {max_retries} attempts, continuing with downloaded data")
                    return all_data
    
    print(f"Downloaded {len(all_data)} records")
    return all_data


# Clean and prepare the data
def clean_and_prepare_data(df):

    print("Cleaning and preparing data:")
    
    # Convert date columns
    date_columns = ['created_date', 'closed_date', 'due_date']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Calculate resolution time in hours
    df['resolution_time_hours'] = (df['closed_date'] - df['created_date']).dt.total_seconds() / 3600
    
    # Remove records without resolution time (still open or invalid dates)
    df = df.dropna(subset=['resolution_time_hours'])
    df = df[df['resolution_time_hours'] >= 0]
    
    # Remove extreme outliers (resolution time > 1 year)
    df = df[df['resolution_time_hours'] <= 365 * 24]
    
    print(f"After cleaning: {len(df)} records")
    
    return df


# Create train/test/validation splits based on created_date
def create_time_splits(df):
  
    print("Creating time-based splits:")
    
    # Sort by created_date
    df = df.sort_values('created_date').reset_index(drop=True)
    
    # Calculate split dates based on data range
    start_date = df['created_date'].min()
    end_date = df['created_date'].max()
    
    print(f"Data range: {start_date} to {end_date}")
    
    # Last 3 months for test & validation
    test_val_start = end_date - timedelta(days=90)  # Last 3 months
    val_start = end_date - timedelta(days=45)  # Last 1.5 months for validation
    
    # Create splits
    train_df = df[df['created_date'] < test_val_start].copy()
    test_df = df[(df['created_date'] >= test_val_start) & (df['created_date'] < val_start)].copy()
    val_df = df[df['created_date'] >= val_start].copy()
    
    print(f"Split dates:")
    print(f"  Train: < {test_val_start}")
    print(f"  Test: {test_val_start} to {val_start}")  
    print(f"  Validation: >= {val_start}")
    print(f"Train set: {len(train_df)} records")
    print(f"Test set: {len(test_df)} records") 
    print(f"Validation set: {len(val_df)} records")
    
    return train_df, test_df, val_df



# Save the processed data
def save_data(train_df, test_df, val_df):
  
    print("Saving processed data:")
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Save datasets
    train_df.to_csv('data/train_data.csv', index=False)
    test_df.to_csv('data/test_data.csv', index=False)
    val_df.to_csv('data/val_data.csv', index=False)
    
    # Save metadata
    metadata = {
        'download_date': datetime.now().isoformat(),
        'train_records': len(train_df),
        'test_records': len(test_df),
        'val_records': len(val_df),
        'total_records': len(train_df) + len(test_df) + len(val_df),
        'columns': list(train_df.columns),
        'date_range': {
            'start': train_df['created_date'].min().isoformat(),
            'end': val_df['created_date'].max().isoformat()
        }
    }
    
    with open('data/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("Data saved successfully!")

def main():

    try:
        # Download data
        all_data = download_311_data()
        
        if not all_data:
            print("No data downloaded!")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame.from_records(all_data)
        print(f"Downloaded {len(df)} records")
        
        # Clean and prepare data
        df_clean = clean_and_prepare_data(df)
        
        # Create time-based splits
        train_df, test_df, val_df = create_time_splits(df_clean)
        
        # Save data
        save_data(train_df, test_df, val_df)
        
        print("\nData download and preprocessing completed successfully!")
        print(f"Files saved in 'data/' directory")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()