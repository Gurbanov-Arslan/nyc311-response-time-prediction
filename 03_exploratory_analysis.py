"""
Full-Dataset Exploratory Data Analysis for NYC 311 Service Requests

Now processes the entire dataset (train + test + validation combined)
and saves all summaries (daily, complaint, borough, etc.) to Excel.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_palette("husl")

os.makedirs('plots', exist_ok=True)
os.makedirs('outputs', exist_ok=True)


# Load and Combine Data
def load_and_merge_data():
    print("Loading preprocessed data...")
    train_df = pd.read_csv('data/train_data.csv')
    test_df = pd.read_csv('data/test_data.csv')
    val_df = pd.read_csv('data/val_data.csv')

    for df in [train_df, test_df, val_df]:
        df['created_date'] = pd.to_datetime(df['created_date'])
        df['closed_date'] = pd.to_datetime(df['closed_date'], errors='coerce')
        if 'due_date' in df.columns:
            df['due_date'] = pd.to_datetime(df['due_date'], errors='coerce')

    full_df = pd.concat([train_df, test_df, val_df], axis=0, ignore_index=True)
    print(f"Combined dataset shape: {full_df.shape}")
    return full_df



# Basic Statistics
def basic_statistics(df):
    print("\n" + "="*60)
    print("BASIC DATASET STATISTICS")
    print("="*60)
    stats = {
        'shape': df.shape,
        'date_min': df['created_date'].min(),
        'date_max': df['created_date'].max(),
        'resolution_time_desc': df['resolution_time_hours'].describe()
    }

    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {stats['date_min']} to {stats['date_max']}")
    print(f"\nResolution Time Statistics:\n{stats['resolution_time_desc'].to_string()}")
    print(f"\nMissing Values:\n{missing.to_string()}")

    return stats, missing


# Resolution Time Analysis
def resolution_time_analysis(df):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Resolution Time Analysis (Full Dataset)', fontsize=16, fontweight='bold')

    axes[0, 0].hist(df['resolution_time_hours'], bins=50, edgecolor='black')
    axes[0, 0].set_xlim(0, np.percentile(df['resolution_time_hours'], 95))
    axes[0, 0].set_title('Distribution of Resolution Times')

    log_resolution = np.log1p(df['resolution_time_hours'])
    axes[0, 1].hist(log_resolution, bins=50, color='orange', edgecolor='black')
    axes[0, 1].set_title('Log-Scale Distribution')

    top_complaints = df['complaint_type'].value_counts().head(8).index
    subset_df = df[df['complaint_type'].isin(top_complaints)]
    subset_df = subset_df[subset_df['resolution_time_hours'] <= np.percentile(subset_df['resolution_time_hours'], 90)]
    sns.boxplot(data=subset_df, y='complaint_type', x='resolution_time_hours', ax=axes[1, 0])
    axes[1, 0].set_title('Resolution Time by Complaint Type')

    daily_avg = df.groupby(df['created_date'].dt.date)['resolution_time_hours'].mean().round(2)
    axes[1, 1].plot(daily_avg.index, daily_avg.values)
    axes[1, 1].set_title('Daily Average Resolution Time')
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('plots/resolution_time_analysis_full.png', dpi=300)
    plt.show()

    return daily_avg


# Complaint Type Analysis
def complaint_type_analysis(df):
    complaint_stats = (
        df.groupby('complaint_type')['resolution_time_hours']
        .agg(['count', 'mean', 'median', 'std'])
        .round(2)
        .sort_values('count', ascending=False)
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Complaint Type Analysis (Full Dataset)', fontsize=16, fontweight='bold')
    top_15 = complaint_stats.head(15)

    axes[0, 0].barh(top_15.index, top_15['count'])
    axes[0, 0].set_title('Top 15 Complaint Types by Volume')

    axes[0, 1].barh(top_15.index, top_15['mean'], color='orange')
    axes[0, 1].set_title('Average Resolution Time by Complaint Type')

    axes[1, 0].scatter(top_15['count'], top_15['mean'])
    axes[1, 0].set_title('Volume vs Resolution Time')

    top_5 = top_15.head(5)
    for complaint in top_5.index:
        subset = df[df['complaint_type'] == complaint]['resolution_time_hours']
        subset = subset[subset <= np.percentile(subset, 95)]
        axes[1, 1].hist(subset, bins=30, alpha=0.6, label=complaint[:15], density=True)
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].set_title('Resolution Time Distribution - Top 5 Complaints')

    plt.tight_layout()
    plt.savefig('plots/complaint_type_analysis_full.png', dpi=300)
    plt.show()

    return complaint_stats


# Temporal Analysis
def temporal_analysis(df):
    df['hour'] = df['created_date'].dt.hour
    df['day_of_week'] = df['created_date'].dt.day_name()
    df['is_weekend'] = df['created_date'].dt.dayofweek >= 5

    hourly_avg = df.groupby('hour')['resolution_time_hours'].mean().round(2)
    day_avg = (
        df.groupby('day_of_week')['resolution_time_hours']
        .mean().reindex(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
        .round(2)
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Temporal Resolution Patterns (Full Dataset)', fontsize=16, fontweight='bold')

    axes[0].plot(hourly_avg.index, hourly_avg.values, marker='o')
    axes[0].set_title('Average Resolution Time by Hour')

    axes[1].bar(day_avg.index, day_avg.values, color='orange')
    axes[1].set_title('Average Resolution Time by Day of Week')

    plt.tight_layout()
    plt.savefig('plots/temporal_analysis_full.png', dpi=300)
    plt.show()

    return hourly_avg, day_avg


# Geographic Analysis
def geographic_analysis(df):
    if 'borough' in df.columns:
        borough_stats = df.groupby('borough')['resolution_time_hours'].agg(['count','mean','median']).round(2)

        plt.figure(figsize=(10,6))
        sns.barplot(x=borough_stats.index, y=borough_stats['mean'])
        plt.title('Avg Resolution Time by Borough (Full Dataset)')
        plt.tight_layout()
        plt.savefig('plots/geographic_analysis_full.png', dpi=300)
        plt.show()

        return borough_stats
    else:
        return pd.DataFrame()



# Correlation Analysis
def correlation_analysis(df):
    num_df = df.select_dtypes(include=['number'])
    corr = num_df.corr(numeric_only=True)['resolution_time_hours'].sort_values(ascending=False)

    plt.figure(figsize=(8,6))
    sns.heatmap(num_df.corr(numeric_only=True), cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap (Full Dataset)')
    plt.tight_layout()
    plt.savefig('plots/correlation_heatmap_full.png', dpi=300)
    plt.show()

    return corr



# Trend Analysis
def trend_analysis(df):
    df['month'] = df['created_date'].dt.to_period('M')
    monthly_stats = df.groupby('month')['resolution_time_hours'].agg(['count','mean','median']).round(2)

    plt.figure(figsize=(12,6))
    plt.plot(monthly_stats.index.astype(str), monthly_stats['mean'], marker='o')
    plt.title('Average Resolution Time by Month (Full Dataset)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/trend_monthly_full.png', dpi=300)
    plt.show()

    return monthly_stats



# Save All Outputs to Excel
def save_analysis_to_excel(stats, missing, corr, daily_avg, complaint_stats,
                           borough_stats, hourly_avg, day_avg, monthly_stats):
    output_path = 'outputs/analysis_outputs_full.xlsx'
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        pd.DataFrame({'Metric': ['Rows','Columns'],
                      'Value': [stats['shape'][0], stats['shape'][1]]}).to_excel(writer, index=False, sheet_name='Basic Stats')
        stats['resolution_time_desc'].to_excel(writer, sheet_name='Resolution Stats')
        missing.to_excel(writer, sheet_name='Missing Values')
        corr.to_frame(name='correlation').to_excel(writer, sheet_name='Correlations')
        daily_avg.to_frame(name='avg_resolution_time').to_excel(writer, sheet_name='Daily Avg Resolution')
        complaint_stats.to_excel(writer, sheet_name='Complaint Type Stats')
        if not borough_stats.empty:
            borough_stats.to_excel(writer, sheet_name='Borough Stats')
        hourly_avg.to_frame(name='avg_resolution_time').to_excel(writer, sheet_name='Hourly Avg Resolution')
        day_avg.to_frame(name='avg_resolution_time').to_excel(writer, sheet_name='DayOfWeek Avg Resolution')
        monthly_stats.to_excel(writer, sheet_name='Monthly Trends')
    print(f" All summaries saved to {output_path}")



def main():
    df = load_and_merge_data()
    stats, missing = basic_statistics(df)

    daily_avg = resolution_time_analysis(df)
    complaint_stats = complaint_type_analysis(df)
    hourly_avg, day_avg = temporal_analysis(df)
    borough_stats = geographic_analysis(df)
    corr = correlation_analysis(df)
    monthly_stats = trend_analysis(df)

    save_analysis_to_excel(stats, missing, corr, daily_avg, complaint_stats,
                           borough_stats, hourly_avg, day_avg, monthly_stats)

    print("\n EDA completed for FULL dataset. Visuals saved in /plots and summaries in /outputs.")

if __name__ == "__main__":
    main()