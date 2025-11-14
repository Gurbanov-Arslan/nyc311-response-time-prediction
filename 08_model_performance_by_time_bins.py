"""
Machine Learning Models for NYC 311 Service Request Resolution Time Prediction

This script implements multiple models to predict resolution time and compares their performance.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Try to import optional dependencies
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    print("Warning: XGBoost not available. Skipping XGBoost model.")
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    print("Warning: LightGBM not available. Skipping LightGBM model.")
    HAS_LIGHTGBM = False


# Load feature-engineered data
def load_feature_data():
    print("Loading feature-engineered data...")
    
    train_df = pd.read_csv('features/train_features.csv')
    test_df = pd.read_csv('features/test_features.csv')
    val_df = pd.read_csv('features/val_features.csv')
    
    with open('features/feature_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    feature_columns = metadata['feature_columns']
    return train_df, test_df, val_df, feature_columns


# Data Preparation
def prepare_data(train_df, test_df, val_df, feature_columns, target='target_log'):
    print(f"Preparing data for modeling with target: {target}")
    
    X_train, y_train = train_df[feature_columns], train_df[target]
    X_test, y_test = test_df[feature_columns], test_df[target]
    X_val, y_val = val_df[feature_columns], val_df[target]
    
    # Clean
    for df in [X_train, X_test, X_val]:
        df.fillna(0, inplace=True)
        df.replace([np.inf, -np.inf], 0, inplace=True)
    
    print(f"Shapes -> Train: {X_train.shape}, Test: {X_test.shape}, Val: {X_val.shape}")
    return X_train, X_test, X_val, y_train, y_test, y_val


def scale_features(X_train, X_test, X_val, method='standard'):
    print(f"Scaling features with {method} scaler")
    scaler = StandardScaler() if method == 'standard' else RobustScaler()
    return (
        scaler.fit_transform(X_train),
        scaler.transform(X_test),
        scaler.transform(X_val),
        scaler
    )


# Calculate evaluation metrics
def calculate_metrics(y_true, y_pred, target_type='log'):
    y_pred = np.clip(y_pred, -50, 50)
    if target_type == 'log':
        y_true_orig, y_pred_orig = np.expm1(y_true), np.expm1(y_pred)
    else:
        y_true_orig, y_pred_orig = y_true, y_pred
    
    y_pred_orig = np.maximum(y_pred_orig, 0)
    y_pred_orig = np.clip(y_pred_orig, 0, 1e6)
    mask = np.isfinite(y_true_orig) & np.isfinite(y_pred_orig)
    y_true_orig, y_pred_orig = y_true_orig[mask], y_pred_orig[mask]
    y_true, y_pred = y_true[mask], y_pred[mask]
    
    return {
        'rmse_orig': np.sqrt(mean_squared_error(y_true_orig, y_pred_orig)),
        'mae_orig': mean_absolute_error(y_true_orig, y_pred_orig),
        'r2_orig': r2_score(y_true_orig, y_pred_orig),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'mape': np.mean(np.abs((y_true_orig - y_pred_orig) / (y_true_orig + 1e-8))) * 100
    }


# Model Trainer Class
class ModelTrainer:
    def __init__(self, X_train, X_test, X_val, y_train, y_test, y_val, target_type='log'):
        self.X_train = X_train
        self.X_test = X_test
        self.X_val = X_val
        self.y_train = y_train
        self.y_test = y_test
        self.y_val = y_val
        self.target_type = target_type
        self.models = {}
        self.results = {}

    def train_random_forest(self):
        print("Training Random Forest...")
        rf = RandomForestRegressor(
            n_estimators=300, max_depth=25, min_samples_split=5, 
            min_samples_leaf=2, random_state=42, n_jobs=-1
        )
        rf.fit(self.X_train, self.y_train)
        self._store_results('Random_Forest', rf)

    def train_gradient_boosting(self):
        print("Training Gradient Boosting...")
        gb = GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=6,
            min_samples_split=5, min_samples_leaf=2, random_state=42
        )
        gb.fit(self.X_train, self.y_train)
        self._store_results('Gradient_Boosting', gb)

    def train_xgboost(self):
        if not HAS_XGBOOST: return
        print("Training XGBoost...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=8,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.5,
            random_state=42, n_jobs=-1, tree_method="hist"
        )
        xgb_model.fit(self.X_train, self.y_train)
        self._store_results('XGBoost', xgb_model)

    def train_lightgbm(self):
        if not HAS_LIGHTGBM: return
        print("Training LightGBM...")
        lgb_model = lgb.LGBMRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=6,
            min_child_samples=20, subsample=0.8,
            colsample_bytree=0.8, random_state=42, n_jobs=-1
        )
        lgb_model.fit(self.X_train, self.y_train)
        self._store_results('LightGBM', lgb_model)

    def _store_results(self, name, model):
        preds = {
            'train': model.predict(self.X_train),
            'test': model.predict(self.X_test),
            'val': model.predict(self.X_val)
        }
        self.models[name] = model
        self.results[name] = {
            'train': calculate_metrics(self.y_train, preds['train'], self.target_type),
            'test': calculate_metrics(self.y_test, preds['test'], self.target_type),
            'val': calculate_metrics(self.y_val, preds['val'], self.target_type),
            'predictions': preds,
            'feature_importance': getattr(model, 'feature_importances_', None)
        }

    def train_all_models(self):
        self.train_random_forest()
        self.train_gradient_boosting()
        self.train_xgboost()
        self.train_lightgbm()
        print("All models trained.")


# Save actual vs predicted values and binned analysis
def save_actual_vs_predicted(trainer, output_dir='models'):
    import os
    os.makedirs(output_dir, exist_ok=True)
    print("Saving actual vs predicted values for all models...")

    for model_name, result in trainer.results.items():
        try:
            y_true = np.expm1(trainer.y_val) if trainer.target_type == 'log' else trainer.y_val
            y_pred = np.expm1(result['predictions']['val']) if trainer.target_type == 'log' else result['predictions']['val']

            df = pd.DataFrame({
                'actual_resolution_time': y_true,
                'predicted_resolution_time': y_pred
            })
            df['error'] = df['predicted_resolution_time'] - df['actual_resolution_time']
            df['abs_error'] = np.abs(df['error'])

            # Create bins
            bins = [0, 1, 6, 12, 24, 48, 72, 168, np.inf]
            labels = ['<1h', '1-6h', '6-12h', '12-24h', '1-2d', '2-3d', '3-7d', '>7d']
            df['actual_bin'] = pd.cut(df['actual_resolution_time'], bins=bins, labels=labels)
            df['predicted_bin'] = pd.cut(df['predicted_resolution_time'], bins=bins, labels=labels)

            # Summary
            summary = df.groupby('actual_bin').agg({
                'actual_resolution_time': 'mean',
                'predicted_resolution_time': 'mean',
                'abs_error': 'mean'
            }).reset_index()

            model_dir = os.path.join(output_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)
            df.to_csv(os.path.join(model_dir, f"{model_name}_actual_vs_predicted_detailed.csv"), index=False)
            summary.to_csv(os.path.join(model_dir, f"{model_name}_binned_summary.csv"), index=False)

            # Plot
            plt.figure(figsize=(8,5))
            plt.bar(summary['actual_bin'], summary['actual_resolution_time'], alpha=0.6, label='Actual')
            plt.bar(summary['actual_bin'], summary['predicted_resolution_time'], alpha=0.6, label='Predicted')
            plt.title(f'{model_name} - Mean Resolution Time by Bin')
            plt.ylabel('Mean Resolution Time (hours)')
            plt.xlabel('Resolution Time Bin')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join('plots', f'{model_name}_binned_performance.png'), dpi=300)
            plt.close()

            print(f"Saved detailed + summary files for {model_name}")
        except Exception as e:
            print(f"Could not save results for {model_name}: {e}")


# MAIN
def main():
    import os
    os.makedirs('plots', exist_ok=True)

    train_df, test_df, val_df, feature_columns = load_feature_data()
    X_train, X_test, X_val, y_train, y_test, y_val = prepare_data(
        train_df, test_df, val_df, feature_columns, target='target_log'
    )
    X_train_scaled, X_test_scaled, X_val_scaled, scaler = scale_features(X_train, X_test, X_val)

    trainer = ModelTrainer(X_train, X_test, X_val, y_train, y_test, y_val)
    trainer.train_all_models()

    summary_df = pd.DataFrame([
        {
            'Model': m,
            'RMSE_orig': r['val']['rmse_orig'],
            'MAE_orig': r['val']['mae_orig'],
            'R2_orig': r['val']['r2_orig'],
            'MAPE': r['val']['mape']
        }
        for m, r in trainer.results.items()
    ])
    summary_df.to_csv('models/model_performance_summary.csv', index=False)
    print(summary_df)

    # Save predictions + binned analysis
    save_actual_vs_predicted(trainer, output_dir='models')


if __name__ == "__main__":
    main()