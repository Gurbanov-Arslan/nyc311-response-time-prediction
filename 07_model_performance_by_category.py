#!/usr/bin/env python3
"""
NYC 311 Service Request Resolution Time Prediction
Model Performance Analysis by Complaint Category
Updated: Tests Random Forest, XGBoost, LightGBM, Gradient Boosting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor


# Calculate performance metrics with error handling for extreme values

def calculate_metrics(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {'rmse': np.inf, 'mae': np.inf, 'r2': -np.inf, 'mape': np.inf, 'count': 0}

    try:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        mape_mask = y_true != 0
        mape = np.mean(np.abs((y_true[mape_mask] - y_pred[mape_mask]) / y_true[mape_mask])) * 100

        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'count': len(y_true)
        }

    except:
        return {'rmse': np.inf, 'mae': np.inf, 'r2': -np.inf, 'mape': np.inf, 'count': len(y_true)}


#  Analyze model performance split by complaint category
def analyze_performance_by_category():

    print("Loading validation data and predictions...")

    val_features = pd.read_csv('features/val_features.csv')
    val_original = pd.read_csv('data/val_data.csv')

    val_data = val_features.copy()
    val_data['complaint_type'] = val_original['complaint_type']

    # Load training
    train_features = pd.read_csv('features/train_features.csv')
    train_original = pd.read_csv('data/train_data.csv')

    exclude_cols = [
        'target_log', 'target', 'target_sqrt', 'target_bin',
        'resolution_time_hours', 'complaint_type',
        'created_date', 'closed_date', 'unique_key'
    ]

    feature_cols = [c for c in val_data.columns if c not in exclude_cols and val_data[c].dtype in ['int64','float64','bool']]

    X_train = train_features[feature_cols]
    y_train = train_features['target_log']

    X_val = val_data[feature_cols]
    y_val_log = val_data['target_log']
    y_val_hours = val_data['resolution_time_hours']
    complaint_types = val_data['complaint_type']

    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()

    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp = imputer.transform(X_val)

    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_val_scaled = scaler.transform(X_val_imp)


    
    # Models to train
    models_to_train = [
        ("Random Forest", RandomForestRegressor(n_estimators=60, n_jobs=-1, random_state=42)),
        ("Gradient Boosting", GradientBoostingRegressor(n_estimators=60, random_state=42)),
        ("XGBoost", XGBRegressor(
            n_estimators=120,
            learning_rate=0.08,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            n_jobs=-1,
            random_state=42
        )),
        ("LightGBM", LGBMRegressor(
            n_estimators=120,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.8,
            random_state=42
        ))
    ]


    # Train + Evaluate models
    results_by_model = {}

    for name, model in models_to_train:
        print(f"\nTraining model: {name}")

        # IF MODEL REQUIRES SCALED INPUTS
        if name in ["Gradient Boosting"]:
            model.fit(X_train_scaled, y_train)
            preds_log = model.predict(X_val_scaled)

        else:  # tree models don't need scaling
            model.fit(X_train_imp, y_train)
            preds_log = model.predict(X_val_imp)

        preds_hours = np.exp(preds_log)

        # Evaluate across top categories
        category_results = {}
        top_categories = complaint_types.value_counts().head(15).index

        for cat in top_categories:
            mask = complaint_types == cat

            metrics = calculate_metrics(
                y_val_hours[mask],
                preds_hours[mask]
            )
            category_results[cat] = metrics

        results_by_model[name] = category_results


   
    # Build Summary file
    summary_rows = []

    for model_name, cat_dict in results_by_model.items():
        for cat, metrics in cat_dict.items():
            summary_rows.append({
                'model': model_name,
                'category': cat,
                'rmse': metrics['rmse'],
                'mae': metrics['mae'],
                'r2': metrics['r2'],
                'mape': metrics['mape'],
                'count': metrics['count']
            })

    summary_df = pd.DataFrame(summary_rows)

    summary_df.to_csv("models/performance_by_category.csv", index=False)

    print("\nSaved performance_by_category.csv")
    print(summary_df.head())

    return summary_df




if __name__ == "__main__":
    print("Running analysis with RF, XGBoost, LightGBM, GradientBoosting...")
    analyze_performance_by_category()
