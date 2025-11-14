#!/usr/bin/env python3
"""
NYC 311 Service Request Resolution Time Prediction
Model Performance Analysis by Complaint Category
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


# Calculate performance metrics with error handling for extreme values

def calculate_metrics(y_true, y_pred):
    
    # Convert to numpy arrays and handle any potential issues
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Remove any infinite or NaN values
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) == 0:
        return {'rmse': np.inf, 'mae': np.inf, 'r2': -np.inf, 'mape': np.inf, 'count': 0}
    
    # Clip extreme values to prevent overflow
    y_pred_clean = np.clip(y_pred_clean, -1e6, 1e6)
    
    try:
        rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
        mae = mean_absolute_error(y_true_clean, y_pred_clean)
        r2 = r2_score(y_true_clean, y_pred_clean)
        
        # Calculate MAPE with protection against division by zero
        mape_mask = y_true_clean != 0
        if np.sum(mape_mask) > 0:
            mape = np.mean(np.abs((y_true_clean[mape_mask] - y_pred_clean[mape_mask]) / y_true_clean[mape_mask])) * 100
        else:
            mape = np.inf
            
        return {
            'rmse': rmse if np.isfinite(rmse) else np.inf,
            'mae': mae if np.isfinite(mae) else np.inf, 
            'r2': r2 if np.isfinite(r2) else -np.inf,
            'mape': mape if np.isfinite(mape) else np.inf,
            'count': len(y_true_clean)
        }
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return {'rmse': np.inf, 'mae': np.inf, 'r2': -np.inf, 'mape': np.inf, 'count': len(y_true_clean)}



#  Analyze model performance split by complaint category
def analyze_performance_by_category():    
    print("Loading validation data and predictions:")
    
    # Load validation features and original data
    val_features = pd.read_csv('features/val_features.csv')
    val_original = pd.read_csv('data/val_data.csv')
    
    # Merge to get complaint types
    val_data = val_features.copy()
    val_data['complaint_type'] = val_original['complaint_type']
    
    # Load model results from JSON files instead of retraining
    models_dir = Path('models')
    results_file = models_dir / 'detailed_results.json'
    
    if not results_file.exists():
        print("No model results found! Please run the machine learning pipeline first.")
        return
    
    print("Training lightweight models for category analysis:")
    
    # Get feature columns (excluding target, categorical, and date columns)
    exclude_cols = ['target_log', 'target', 'target_sqrt', 'target_bin', 'resolution_time_hours', 'complaint_type', 
                   'created_date', 'closed_date', 'unique_key']
    feature_cols = [col for col in val_data.columns if col not in exclude_cols and val_data[col].dtype in ['int64', 'float64', 'bool']]
    
    # Load training data for model training
    train_features = pd.read_csv('features/train_features.csv')
    train_original = pd.read_csv('data/train_data.csv')
    
    # Prepare training data
    X_train = train_features[feature_cols]
    y_train = train_features['target_log']  # Use log-transformed target
    
    # Prepare validation data
    X_val = val_data[feature_cols]
    y_val_log = val_data['target_log']
    y_val_hours = val_data['resolution_time_hours']  # Use available resolution time column
    complaint_types = val_data['complaint_type']
    
    # Handle missing values
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    
    # Fill missing values with median
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_val_imputed = imputer.transform(X_val)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_val_scaled = scaler.transform(X_val_imputed)
    
    # Train lightweight versions of best models
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    
    models_to_train = [
        ('Random Forest', RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)),
        ('Gradient Boosting', GradientBoostingRegressor(n_estimators=50, random_state=42)),
        ('Ridge Regression', Ridge(alpha=1.0))
    ]
    
    # Train and analyze each model
    results_by_model = {}
    
    for model_name, model in models_to_train:
        print(f"\nTraining and analyzing {model_name}...")
        
        try:
            # Train model
            if model_name == 'Ridge Regression':
                model.fit(X_train_scaled, y_train)
                y_pred_log = model.predict(X_val_scaled)
            else:
                model.fit(X_train_imputed, y_train)
                y_pred_log = model.predict(X_val_imputed)
            
            # Convert back to original scale
            y_pred = np.exp(y_pred_log)
            
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            continue
        
        # Calculate performance by complaint category
        category_performance = {}
        unique_categories = complaint_types.value_counts().head(15).index  # Top 15 categories
        
        for category in unique_categories:
            mask = complaint_types == category
            if mask.sum() < 10:  # Skip categories with too few samples
                continue
            
            y_true_cat = y_val_hours[mask]
            y_pred_cat = y_pred[mask]
            
            metrics = calculate_metrics(y_true_cat, y_pred_cat)
            category_performance[category] = metrics
        
        results_by_model[model_name] = category_performance
    
    # Create comprehensive analysis
    print("\n" + "="*80)
    print("MODEL PERFORMANCE BY COMPLAINT CATEGORY ANALYSIS")
    print("="*80)
    
    # Create summary dataframe
    summary_data = []
    for model_name, categories in results_by_model.items():
        for category, metrics in categories.items():
            summary_data.append({
                'model': model_name,
                'category': category,
                'count': metrics['count'],
                'rmse': metrics['rmse'],
                'mae': metrics['mae'],
                'r2': metrics['r2'],
                'mape': metrics['mape']
            })
    
    if not summary_data:
        print("No performance data generated. Check your data and models.")
        return
    
    summary_df = pd.DataFrame(summary_data)
    
    # Overall category statistics
    print("\nComplaint Category Statistics:")
    print("-" * 50)
    category_stats = complaint_types.value_counts().head(15)
    for category, count in category_stats.items():
        pct = (count / len(complaint_types)) * 100
        print(f"{category:30s}: {count:6d} ({pct:5.1f}%)")
    
    # Best and worst performing categories for each model
    for model_name in results_by_model.keys():
        if model_name not in results_by_model:
            continue
            
        print(f"\n{model_name.upper()} - Performance by Category:")
        print("-" * 60)
        
        model_data = summary_df[summary_df['model'] == model_name].copy()
        if len(model_data) == 0:
            continue
            
        # Sort by R² score (descending)
        model_data = model_data.sort_values('r2', ascending=False)
        
        print("Best Performing Categories (Top 5):")
        for _, row in model_data.head(5).iterrows():
            print(f"  {row['category']:25s}: R²={row['r2']:6.3f}, RMSE={row['rmse']:7.1f}h, Count={row['count']:4d}")
        
        print("Worst Performing Categories (Bottom 5):")
        for _, row in model_data.tail(5).iterrows():
            print(f"  {row['category']:25s}: R²={row['r2']:6.3f}, RMSE={row['rmse']:7.1f}h, Count={row['count']:4d}")
    
    # Create visualizations
    print(f"\nCreating visualizations...")
    
    # Ensure plots directory exists
    plots_dir = Path('plots')
    plots_dir.mkdir(exist_ok=True)
    
    # 1. Performance comparison heatmap
    plt.figure(figsize=(15, 10))
    
    # Create pivot table for heatmap
    pivot_r2 = summary_df.pivot(index='category', columns='model', values='r2')
    pivot_r2 = pivot_r2.fillna(-999)  # Fill NaN with very low value for visualization
    
    # Sort categories by average R² across models
    category_avg_r2 = pivot_r2.mean(axis=1).sort_values(ascending=False)
    pivot_r2 = pivot_r2.loc[category_avg_r2.index]
    
    sns.heatmap(pivot_r2, annot=True, cmap='RdYlBu_r', center=0, 
                fmt='.3f', cbar_kws={'label': 'R² Score'})
    plt.title('Model Performance (R²) by Complaint Category', fontsize=16, fontweight='bold')
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Complaint Category', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(plots_dir / 'performance_by_category_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. RMSE comparison by category (top categories only)
    top_categories = category_stats.head(10).index
    top_data = summary_df[summary_df['category'].isin(top_categories)]
    
    plt.figure(figsize=(15, 8))
    pivot_rmse = top_data.pivot(index='category', columns='model', values='rmse')
    
    # Create grouped bar plot
    ax = pivot_rmse.plot(kind='bar', figsize=(15, 8), width=0.8)
    plt.title('RMSE by Complaint Category (Top 10 Categories)', fontsize=16, fontweight='bold')
    plt.xlabel('Complaint Category', fontsize=12)
    plt.ylabel('RMSE (hours)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / 'rmse_by_category_top10.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. R² distribution across categories
    plt.figure(figsize=(12, 8))
    
    # Create box plot for R² distribution by model
    filtered_data = summary_df[summary_df['r2'] > -10]  # Filter out extreme outliers
    
    sns.boxplot(data=filtered_data, x='model', y='r2')
    plt.title('R² Score Distribution Across Categories by Model', fontsize=16, fontweight='bold')
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('R² Score', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / 'r2_distribution_by_model.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Category difficulty analysis
    plt.figure(figsize=(14, 10))
    
    # Calculate average performance across all models for each category
    category_avg_performance = summary_df.groupby('category').agg({
        'r2': 'mean',
        'rmse': 'mean',
        'count': 'first'
    }).reset_index()
    
    # Sort by R² and take top 15
    category_avg_performance = category_avg_performance.sort_values('r2', ascending=True).tail(15)
    
    # Create horizontal bar plot
    plt.barh(range(len(category_avg_performance)), category_avg_performance['r2'], 
             color=['red' if x < 0 else 'green' for x in category_avg_performance['r2']])
    plt.yticks(range(len(category_avg_performance)), category_avg_performance['category'].tolist())
    plt.xlabel('Average R² Score Across All Models', fontsize=12)
    plt.title('Complaint Categories by Prediction Difficulty\n(Lower R² = Harder to Predict)', 
              fontsize=16, fontweight='bold')
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.7)
    plt.grid(True, alpha=0.3, axis='x')
    
    # Add count annotations
    for i, (idx, row) in enumerate(category_avg_performance.iterrows()):
        plt.text(row['r2'] + 0.01 if row['r2'] >= 0 else row['r2'] - 0.01, i, 
                f"n={int(row['count'])}", va='center', 
                ha='left' if row['r2'] >= 0 else 'right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'category_difficulty_ranking.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed results to CSV
    summary_df.to_csv('models/performance_by_category.csv', index=False)
    
    # Create insights summary
    insights = generate_category_insights(summary_df, category_stats)
    
    # Save insights to file
    with open('models/category_performance_insights.txt', 'w') as f:
        f.write(insights)
    
    print(f"\nAnalysis complete!")
    print(f" Visualizations saved in plots/ directory:")
    print(f"   • performance_by_category_heatmap.png")
    print(f"   • rmse_by_category_top10.png") 
    print(f"   • r2_distribution_by_model.png")
    print(f"   • category_difficulty_ranking.png")
    print(f" Detailed results saved to models/performance_by_category.csv")
    print(f" Insights saved to models/category_performance_insights.txt")
    
    return summary_df

def generate_category_insights(summary_df, category_stats):
    
    # Initialize insights string
    insights = """Executive Summary"""
    
    # Find best and worst categories overall
    category_avg = summary_df.groupby('category')['r2'].mean().sort_values(ascending=False)
    best_categories = category_avg.head(5)
    worst_categories = category_avg.tail(5)
    
    insights += f"\n1. MOST PREDICTABLE COMPLAINT TYPES:\n"
    for category, r2 in best_categories.items():
        count = category_stats.get(category, 0)
        insights += f"   • {category}: R² = {r2:.3f} (n={count:,})\n"
    
    insights += f"\n2. LEAST PREDICTABLE COMPLAINT TYPES:\n"
    for category, r2 in worst_categories.items():
        count = category_stats.get(category, 0)
        insights += f"   • {category}: R² = {r2:.3f} (n={count:,})\n"
    
    # Model consistency analysis
    insights += f"\n3. MODEL CONSISTENCY ACROSS CATEGORIES:\n"
    for model in summary_df['model'].unique():
        model_data = summary_df[summary_df['model'] == model]
        avg_r2 = model_data['r2'].mean()
        std_r2 = model_data['r2'].std()
        insights += f"   • {model.replace('_', ' ').title()}: Avg R² = {avg_r2:.3f} (±{std_r2:.3f})\n"
    
    # Volume vs Performance analysis
    high_volume_categories = category_stats.head(10).index
    high_vol_performance = summary_df[summary_df['category'].isin(high_volume_categories)]['r2'].mean()
    low_vol_performance = summary_df[~summary_df['category'].isin(high_volume_categories)]['r2'].mean()
    
    insights += f"\nVOLUME vs PREDICTABILITY:\n"
    insights += f"   • High-volume categories (top 10): Avg R² = {high_vol_performance:.3f}\n"
    insights += f"   • Low-volume categories: Avg R² = {low_vol_performance:.3f}\n"
   
    return insights

if __name__ == "__main__":
    print("Starting Model Performance by Category Analysis...")
    analyze_performance_by_category()