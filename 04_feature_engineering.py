"""
Feature Engineering for NYC 311 Service Requests

This script creates features for predicting resolution time based on the EDA insights.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


#  Load the preprocessed data
def load_data():

    print("Loading data for feature engineering:")
    
    train_df = pd.read_csv('data/train_data.csv')
    test_df = pd.read_csv('data/test_data.csv')
    val_df = pd.read_csv('data/val_data.csv')
    
    # Convert date columns
    for df in [train_df, test_df, val_df]:
        df['created_date'] = pd.to_datetime(df['created_date'])
        df['closed_date'] = pd.to_datetime(df['closed_date'])
        if 'due_date' in df.columns:
            df['due_date'] = pd.to_datetime(df['due_date'])
    
    return train_df, test_df, val_df



# Create temporal features from created_date
def create_temporal_features(df):

    print("Creating temporal features:")
    
    # Basic time components
    df['hour'] = df['created_date'].dt.hour
    df['day_of_week'] = df['created_date'].dt.dayofweek
    df['day_of_month'] = df['created_date'].dt.day
    df['month'] = df['created_date'].dt.month
    df['quarter'] = df['created_date'].dt.quarter
    df['year'] = df['created_date'].dt.year
    
    # Business hours and weekend indicators
    df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_monday'] = (df['day_of_week'] == 0).astype(int)
    df['is_friday'] = (df['day_of_week'] == 4).astype(int)
    
    # Time-based bins (encode as dummy variables instead of categorical)
    hour_bins = pd.cut(df['hour'], bins=[0, 6, 12, 18, 24], labels=['Night', 'Morning', 'Afternoon', 'Evening'])
    day_bins = pd.cut(df['day_of_month'], bins=[0, 10, 20, 31], labels=['Early', 'Mid', 'Late'])
    
    # Convert to dummy variables
    hour_dummies = pd.get_dummies(hour_bins, prefix='hour_bin')
    day_dummies = pd.get_dummies(day_bins, prefix='day_bin')
    
    df = pd.concat([df, hour_dummies, day_dummies], axis=1)
    
    # Cyclical encoding for temporal features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    return df



# Create features related to complaint types and descriptors
def create_complaint_features(df, train_df):

    print("Creating complaint-related features:")
    
    # Complaint type frequency encoding (based on training data)
    complaint_freq = train_df['complaint_type'].value_counts().to_dict()
    df['complaint_frequency'] = df['complaint_type'].map(complaint_freq).fillna(0)
    
    # Average resolution time by complaint type (from training data)
    complaint_avg_resolution = train_df.groupby('complaint_type')['resolution_time_hours'].mean().to_dict()
    df['complaint_avg_resolution'] = df['complaint_type'].map(complaint_avg_resolution)
    
    # Complaint type categories based on average resolution time
    if 'complaint_avg_resolution' in df.columns:
        urgency_bins = pd.cut(df['complaint_avg_resolution'], 
                             bins=[0, 10, 100, 1000, float('inf')],
                             labels=['Urgent', 'Medium', 'Long', 'Very_Long'])
        urgency_dummies = pd.get_dummies(urgency_bins, prefix='complaint_urgency')
        df = pd.concat([df, urgency_dummies], axis=1)
    
    # Top complaint type flags
    top_complaints = train_df['complaint_type'].value_counts().head(15).index
    for complaint in top_complaints:
        safe_name = complaint.replace(' ', '_').replace('/', '_').replace('-', '_')
        df[f'is_{safe_name}'] = (df['complaint_type'] == complaint).astype(int)
    
    # Descriptor length and word count
    df['descriptor_length'] = df['descriptor'].fillna('').astype(str).str.len()
    df['descriptor_word_count'] = df['descriptor'].fillna('').astype(str).str.split().str.len()
    
    return df




# Create geographic-related features
def create_geographic_features(df, train_df):
    
    print("Creating geographic features:")

    # Borough frequency and average resolution time
    if 'borough' in df.columns:
        borough_freq = train_df['borough'].value_counts().to_dict()
        df['borough_frequency'] = df['borough'].map(borough_freq).fillna(0)
        
        borough_avg_resolution = train_df.groupby('borough')['resolution_time_hours'].mean().to_dict()
        df['borough_avg_resolution'] = df['borough'].map(borough_avg_resolution)
        
        # One-hot encode borough
        borough_dummies = pd.get_dummies(df['borough'], prefix='borough')
        df = pd.concat([df, borough_dummies], axis=1)
    
    # Zip code features
    if 'incident_zip' in df.columns:
        # Clean zip codes
        df['incident_zip'] = df['incident_zip'].fillna('00000').astype(str).str[:5]
        
        # Zip code frequency
        zip_freq = train_df['incident_zip'].value_counts().to_dict()
        df['zip_frequency'] = df['incident_zip'].map(zip_freq).fillna(0)
        
        # High-frequency zip code indicator
        top_zips = train_df['incident_zip'].value_counts().head(20).index
        df['is_top_zip'] = df['incident_zip'].isin(top_zips).astype(int)
    
    # Location type features
    if 'location_type' in df.columns:
        location_freq = train_df['location_type'].value_counts().to_dict()
        df['location_type_frequency'] = df['location_type'].map(location_freq).fillna(0)
        
        # Top location types
        top_locations = train_df['location_type'].value_counts().head(10).index
        for location in top_locations:
            if pd.notna(location):
                safe_name = str(location).replace(' ', '_').replace('/', '_').replace('-', '_')
                df[f'is_location_{safe_name}'] = (df['location_type'] == location).astype(int)
    
    return df



# Create agency-related features
def create_agency_features(df, train_df):
    
    print("Creating agency features:")
    
    # Agency frequency and average resolution time
    agency_freq = train_df['agency'].value_counts().to_dict()
    df['agency_frequency'] = df['agency'].map(agency_freq).fillna(0)
    
    agency_avg_resolution = train_df.groupby('agency')['resolution_time_hours'].mean().to_dict()
    df['agency_avg_resolution'] = df['agency'].map(agency_avg_resolution)
    
    # Top agencies
    top_agencies = train_df['agency'].value_counts().head(10).index
    for agency in top_agencies:
        safe_name = agency.replace(' ', '_').replace('/', '_').replace('-', '_')
        df[f'is_agency_{safe_name}'] = (df['agency'] == agency).astype(int)
    
    return df



#  Create interaction features
def create_interaction_features(df):

    print("Creating interaction features:")
    
    # Complaint type × Time interactions
    df['complaint_freq_x_weekend'] = df['complaint_frequency'] * df['is_weekend']
    df['complaint_freq_x_business_hours'] = df['complaint_frequency'] * df['is_business_hours']
    
    # Borough × Time interactions
    if 'borough_frequency' in df.columns:
        df['borough_freq_x_weekend'] = df['borough_frequency'] * df['is_weekend']
        df['borough_freq_x_business_hours'] = df['borough_frequency'] * df['is_business_hours']
    
    # Agency × Complaint interactions
    df['agency_freq_x_complaint_freq'] = df['agency_frequency'] * df['complaint_frequency']
    
    return df




#  Create features from text columns

def create_text_features(df, train_df, max_features=100):
    print("Creating text features:")
    
    # Simple text-based features instead of TF-IDF for now
    text_columns = ['descriptor', 'resolution_description']
    
    # Text length and word count features
    for col in text_columns:
        if col in df.columns:
            df[f'{col}_length'] = df[col].fillna('').astype(str).str.len()
            df[f'{col}_word_count'] = df[col].fillna('').astype(str).str.split().str.len()
            df[f'{col}_has_text'] = (df[col].fillna('').astype(str).str.len() > 0).astype(int)
    
    # Combined text features
    combined_text = []
    for idx, row in df.iterrows():
        text_parts = []
        for col in text_columns:
            if col in df.columns and pd.notna(row[col]):
                text_parts.append(str(row[col]))
        combined_text.append(' '.join(text_parts))
    
    df['combined_text_length'] = [len(text) for text in combined_text]
    df['combined_text_word_count'] = [len(text.split()) for text in combined_text]
    
    return df


# Create target variable and variations

def create_target_variable(df):

    print("Creating target variables:")
    
    # Original target
    df['target'] = df['resolution_time_hours']
    
    # Log-transformed target (to handle skewness)
    df['target_log'] = np.log1p(df['resolution_time_hours'])
    
    # Square root transformed target
    df['target_sqrt'] = np.sqrt(df['resolution_time_hours'])
    
    # Binned target for classification approaches
    df['target_bin'] = pd.cut(df['resolution_time_hours'], 
                             bins=[0, 1, 24, 72, 168, float('inf')],
                             labels=['Very_Fast', 'Fast', 'Medium', 'Slow', 'Very_Slow'])
    
    return df



# Select relevant features for modeling

def select_features(df, target_col='target'):

    print("Selecting features for modeling:")
    
    # Define feature categories
    temporal_features = [col for col in df.columns if any(x in col for x in ['hour', 'day', 'month', 'quarter', 'year', 'weekend', 'business', 'sin', 'cos'])]
    complaint_features = [col for col in df.columns if any(x in col for x in ['complaint', 'descriptor', 'urgency', 'is_HEAT', 'is_Illegal', 'is_Noise', 'is_Blocked', 'is_UNSANITARY'])]
    geographic_features = [col for col in df.columns if any(x in col for x in ['borough', 'zip', 'location'])]
    agency_features = [col for col in df.columns if 'agency' in col]
    interaction_features = [col for col in df.columns if '_x_' in col]
    text_features = [col for col in df.columns if 'tfidf' in col]
    
    # Combine all feature categories
    feature_columns = (temporal_features + complaint_features + geographic_features + 
                      agency_features + interaction_features + text_features)
    
    # Remove duplicates and ensure columns exist
    feature_columns = list(set(feature_columns))
    feature_columns = [col for col in feature_columns if col in df.columns]
    
    # Exclude target-related columns from features to prevent data leakage
    target_related_columns = ['resolution_time_hours', 'target', 'target_log', 'target_sqrt', 'target_bin']
    feature_columns = [col for col in feature_columns if col not in target_related_columns]
    
    # Filter to keep only numeric columns
    numeric_feature_columns = []
    for col in feature_columns:
        if df[col].dtype in ['int64', 'float64', 'bool', 'int32', 'float32']:
            numeric_feature_columns.append(col)
        elif col.startswith('is_') or col.startswith('borough_') or col.startswith('hour_bin_') or col.startswith('day_bin_') or col.startswith('complaint_urgency_'):
            # These should be numeric/dummy variables
            if df[col].dtype == 'object':
                try:
                    pd.to_numeric(df[col], errors='raise')
                    numeric_feature_columns.append(col)
                except:
                    print(f"Skipping non-numeric column: {col}")
            else:
                numeric_feature_columns.append(col)
        else:
            print(f"Skipping non-numeric column: {col}")
    
    feature_columns = numeric_feature_columns
    
    # Add target columns (keep resolution_time_hours only for reference, not as feature)
    target_columns = ['target', 'target_log', 'target_sqrt', 'target_bin', 'resolution_time_hours']
    target_columns = [col for col in target_columns if col in df.columns]
    
    # Select final columns
    final_columns = feature_columns + target_columns + ['unique_key', 'created_date', 'closed_date']
    final_columns = [col for col in final_columns if col in df.columns]
    
    print(f"Selected {len(feature_columns)} numeric features for modeling:")
    temporal_count = len([col for col in feature_columns if any(x in col for x in ['hour', 'day', 'month', 'quarter', 'year', 'weekend', 'business', 'sin', 'cos'])])
    complaint_count = len([col for col in feature_columns if any(x in col for x in ['complaint', 'descriptor', 'urgency', 'is_HEAT', 'is_Illegal', 'is_Noise', 'is_Blocked', 'is_UNSANITARY'])])
    geographic_count = len([col for col in feature_columns if any(x in col for x in ['borough', 'zip', 'location'])])
    agency_count = len([col for col in feature_columns if 'agency' in col])
    interaction_count = len([col for col in feature_columns if '_x_' in col])
    text_count = len([col for col in feature_columns if 'tfidf' in col])
    
    print(f"  Temporal: {temporal_count}")
    print(f"  Complaint: {complaint_count}")
    print(f"  Geographic: {geographic_count}")
    print(f"  Agency: {agency_count}")
    print(f"  Interaction: {interaction_count}")
    print(f"  Text: {text_count}")
    
    return df[final_columns], feature_columns



# Save the feature-engineered data
def save_engineered_data(train_df, test_df, val_df, feature_columns):
    print("Saving feature-engineered data:")
    
    # Create features directory
    import os
    os.makedirs('features', exist_ok=True)
    
    # Save datasets
    train_df.to_csv('features/train_features.csv', index=False)
    test_df.to_csv('features/test_features.csv', index=False)
    val_df.to_csv('features/val_features.csv', index=False)
    
    # Save feature metadata
    feature_metadata = {
        'feature_engineering_date': datetime.now().isoformat(),
        'total_features': len(feature_columns),
        'feature_columns': feature_columns,
        'train_shape': train_df.shape,
        'test_shape': test_df.shape,
        'val_shape': val_df.shape,
        'target_variables': ['target', 'target_log', 'target_sqrt', 'target_bin']
    }
    
    import json
    with open('features/feature_metadata.json', 'w') as f:
        json.dump(feature_metadata, f, indent=2)
    
    print("Feature-engineered data saved successfully!")


#  Main feature engineering pipeline
def main():
  
    try:
        # Load data
        train_df, test_df, val_df = load_data()
        
        # Feature engineering pipeline
        print("\nStarting feature engineering pipeline:")
        
        datasets = []
        for name, df in [('Train', train_df), ('Test', test_df), ('Validation', val_df)]:
            print(f"\nProcessing {name} dataset:")
            
            # Create features
            df = create_temporal_features(df)
            df = create_complaint_features(df, train_df)  # Always use train_df for encoding
            df = create_geographic_features(df, train_df)
            df = create_agency_features(df, train_df)
            df = create_interaction_features(df)
            df = create_text_features(df, train_df)
            df = create_target_variable(df)
            
            datasets.append(df)
        
        train_featured, test_featured, val_featured = datasets
        
        # Select features
        train_final, feature_columns = select_features(train_featured)
        test_final, _ = select_features(test_featured)
        val_final, _ = select_features(val_featured)
        
        # Ensure all datasets have the same columns
        common_columns = set(train_final.columns) & set(test_final.columns) & set(val_final.columns)
        train_final = train_final[list(common_columns)]
        test_final = test_final[list(common_columns)]
        val_final = val_final[list(common_columns)]
        
        # Update feature columns list
        feature_columns = [col for col in feature_columns if col in common_columns]
        
        # Save engineered data
        save_engineered_data(train_final, test_final, val_final, feature_columns)
        
        print(f"\nFeature engineering completed successfully!")
        print(f"Final dataset shapes:")
        print(f"  Train: {train_final.shape}")
        print(f"  Test: {test_final.shape}")
        print(f"  Validation: {val_final.shape}")
        print(f"  Features for modeling: {len(feature_columns)}")
        
    except Exception as e:
        print(f"Error in feature engineering: {e}")
        raise

if __name__ == "__main__":
    main()