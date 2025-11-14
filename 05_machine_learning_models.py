"""
Machine Learning Models for NYC 311 Service Request Resolution Time Prediction

This script implements multiple models to predict resolution time and compares their performance.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
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

def load_feature_data():
    """
    Load the feature-engineered data
    """
    print("Loading feature-engineered data...")
    
    train_df = pd.read_csv('features/train_features.csv')
    test_df = pd.read_csv('features/test_features.csv')
    val_df = pd.read_csv('features/val_features.csv')
    
    # Load feature metadata
    with open('features/feature_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    feature_columns = metadata['feature_columns']
    
    return train_df, test_df, val_df, feature_columns


# Prepare data for modeling
def prepare_data(train_df, test_df, val_df, feature_columns, target='target_log'):

    print(f"Preparing data for modeling with target: {target}")
    
    # Separate features and target
    X_train = train_df[feature_columns].copy()
    y_train = train_df[target].copy()
    
    X_test = test_df[feature_columns].copy()
    y_test = test_df[target].copy()
    
    X_val = val_df[feature_columns].copy()
    y_val = val_df[target].copy()
    
    # Handle missing values
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    X_val = X_val.fillna(0)
    
    # Handle infinite values
    X_train = X_train.replace([np.inf, -np.inf], 0)
    X_test = X_test.replace([np.inf, -np.inf], 0)
    X_val = X_val.replace([np.inf, -np.inf], 0)
    
    print(f"Data shapes:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")
    print(f"  X_val: {X_val.shape}, y_val: {y_val.shape}")
    
    return X_train, X_test, X_val, y_train, y_test, y_val


# Scale features
def scale_features(X_train, X_test, X_val, method='standard'):

    print(f"Scaling features using {method} scaling:")
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        return X_train, X_test, X_val, None
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_val_scaled = scaler.transform(X_val)
    
    return X_train_scaled, X_test_scaled, X_val_scaled, scaler


# Calculate evaluation metrics
def calculate_metrics(y_true, y_pred, target_type='log'):
 
    # Handle extreme values in predictions
    y_pred = np.clip(y_pred, -50, 50)  # Reasonable range for log-transformed values
    
    # If target is log-transformed, convert back to original scale
    if target_type == 'log':
        y_true_orig = np.expm1(y_true)
        y_pred_orig = np.expm1(y_pred)
    elif target_type == 'sqrt':
        y_true_orig = np.square(y_true)
        y_pred_orig = np.square(y_pred)
    else:
        y_true_orig = y_true
        y_pred_orig = y_pred
    
    # Ensure no negative predictions and handle extreme values
    y_pred_orig = np.maximum(y_pred_orig, 0)
    y_pred_orig = np.clip(y_pred_orig, 0, 1e6)  # Cap at 1M hours to avoid overflow
    
    # Handle infinite and NaN values
    finite_mask = np.isfinite(y_pred_orig) & np.isfinite(y_true_orig)
    y_pred_orig = y_pred_orig[finite_mask]
    y_true_orig = y_true_orig[finite_mask]
    y_pred_finite = y_pred[finite_mask]
    y_true_finite = y_true[finite_mask]
    
    try:
        metrics = {
            'mse': mean_squared_error(y_true_finite, y_pred_finite),
            'rmse': np.sqrt(mean_squared_error(y_true_finite, y_pred_finite)),
            'mae': mean_absolute_error(y_true_finite, y_pred_finite),
            'r2': r2_score(y_true_finite, y_pred_finite),
            'mse_orig': mean_squared_error(y_true_orig, y_pred_orig),
            'rmse_orig': np.sqrt(mean_squared_error(y_true_orig, y_pred_orig)),
            'mae_orig': mean_absolute_error(y_true_orig, y_pred_orig),
            'r2_orig': r2_score(y_true_orig, y_pred_orig),
            'mape': np.mean(np.abs((y_true_orig - y_pred_orig) / (y_true_orig + 1e-8))) * 100
        }
    except Exception as e:
        print(f"Warning: Error calculating metrics: {e}")
        # Return default metrics if calculation fails
        metrics = {
            'mse': float('inf'), 'rmse': float('inf'), 'mae': float('inf'), 'r2': -1,
            'mse_orig': float('inf'), 'rmse_orig': float('inf'), 'mae_orig': float('inf'), 'r2_orig': -1,
            'mape': float('inf')
        }
    
    return metrics



# Class to train and evaluate models
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
    
  
    # Random Forest
    def train_random_forest(self):
        
        print("Training Random Forest:")
        
        # Base Random Forest
        rf = RandomForestRegressor(
          n_estimators=300,
            max_depth=25,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        rf.fit(self.X_train, self.y_train)
        
        # Predictions
        y_train_pred = rf.predict(self.X_train)
        y_test_pred = rf.predict(self.X_test)
        y_val_pred = rf.predict(self.X_val)
        
        # Metrics
        train_metrics = calculate_metrics(self.y_train, y_train_pred, self.target_type)
        test_metrics = calculate_metrics(self.y_test, y_test_pred, self.target_type)
        val_metrics = calculate_metrics(self.y_val, y_val_pred, self.target_type)
        
        self.models['Random_Forest'] = rf
        self.results['Random_Forest'] = {
            'train': train_metrics,
            'test': test_metrics,
            'val': val_metrics,
            'predictions': {
                'train': y_train_pred,
                'test': y_test_pred,
                'val': y_val_pred
            },
            'feature_importance': rf.feature_importances_
        }
    
    # Gradient Boosting
    def train_gradient_boosting(self):
    
        print("Training Gradient Boosting models:")
        
        # Scikit-learn Gradient Boosting
        gb = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        gb.fit(self.X_train, self.y_train)
        
        # Predictions
        y_train_pred = gb.predict(self.X_train)
        y_test_pred = gb.predict(self.X_test)
        y_val_pred = gb.predict(self.X_val)
        
        # Metrics
        train_metrics = calculate_metrics(self.y_train, y_train_pred, self.target_type)
        test_metrics = calculate_metrics(self.y_test, y_test_pred, self.target_type)
        val_metrics = calculate_metrics(self.y_val, y_val_pred, self.target_type)
        
        self.models['Gradient_Boosting'] = gb
        self.results['Gradient_Boosting'] = {
            'train': train_metrics,
            'test': test_metrics,
            'val': val_metrics,
            'predictions': {
                'train': y_train_pred,
                'test': y_test_pred,
                'val': y_val_pred
            },
            'feature_importance': gb.feature_importances_
        }
    # XGBoost
    def train_xgboost(self):
       
        if not HAS_XGBOOST:
            print("Skipping XGBoost - library not available")
            return
            
        print("Training XGBoost:")
        import xgboost as xgb
        
        xgb_model = xgb.XGBRegressor(
         n_estimators=300,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.5,
            random_state=42,
            n_jobs=-1,
            tree_method="hist"
        )
        
        xgb_model.fit(self.X_train, self.y_train)
        
        # Predictions
        y_train_pred = xgb_model.predict(self.X_train)
        y_test_pred = xgb_model.predict(self.X_test)
        y_val_pred = xgb_model.predict(self.X_val)
        
        # Metrics
        train_metrics = calculate_metrics(self.y_train, y_train_pred, self.target_type)
        test_metrics = calculate_metrics(self.y_test, y_test_pred, self.target_type)
        val_metrics = calculate_metrics(self.y_val, y_val_pred, self.target_type)
        
        self.models['XGBoost'] = xgb_model
        self.results['XGBoost'] = {
            'train': train_metrics,
            'test': test_metrics,
            'val': val_metrics,
            'predictions': {
                'train': y_train_pred,
                'test': y_test_pred,
                'val': y_val_pred
            },
            'feature_importance': xgb_model.feature_importances_
        }
    # LightGBM
    def train_lightgbm(self):
    
        if not HAS_LIGHTGBM:
            print("Skipping LightGBM - library not available")
            return
            
        print("Training LightGBM:")
        import lightgbm as lgb
        
        lgb_model = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        lgb_model.fit(self.X_train, self.y_train)
        
        # Predictions
        y_train_pred = lgb_model.predict(self.X_train)
        y_test_pred = lgb_model.predict(self.X_test)
        y_val_pred = lgb_model.predict(self.X_val)
        
        # Metrics
        train_metrics = calculate_metrics(self.y_train, y_train_pred, self.target_type)
        test_metrics = calculate_metrics(self.y_test, y_test_pred, self.target_type)
        val_metrics = calculate_metrics(self.y_val, y_val_pred, self.target_type)
        
        self.models['LightGBM'] = lgb_model
        self.results['LightGBM'] = {
            'train': train_metrics,
            'test': test_metrics,
            'val': val_metrics,
            'predictions': {
                'train': y_train_pred,
                'test': y_test_pred,
                'val': y_val_pred
            },
            'feature_importance': lgb_model.feature_importances_
        }

    
    def train_all_models(self):
        
        print("Training all models:")
        
        # self.train_linear_regression()
        self.train_random_forest()
        self.train_gradient_boosting()
        self.train_xgboost()
        self.train_lightgbm()
        # self.train_neural_network()
        
        print("All models trained successfully!")


# Create results summary
def create_results_summary(results):

    print("\n" + "="*80)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*80)
    
    # Create summary DataFrame
    summary_data = []
    
    for model_name, result in results.items():
        for dataset in ['train', 'test', 'val']:
            metrics = result[dataset]
            summary_data.append({
                'Model': model_name,
                'Dataset': dataset,
                'RMSE_orig': metrics['rmse_orig'],
                'MAE_orig': metrics['mae_orig'],
                'R2_orig': metrics['r2_orig'],
                'MAPE': metrics['mape'],
                'RMSE_log': metrics['rmse'],
                'R2_log': metrics['r2']
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Display validation results
    val_results = summary_df[summary_df['Dataset'] == 'val'].sort_values('RMSE_orig')
    print("\nValidation Set Performance (sorted by RMSE on original scale):")
    print("-" * 80)
    for _, row in val_results.iterrows():
        print(f"{row['Model']:<20} | RMSE: {row['RMSE_orig']:>8.2f} | MAE: {row['MAE_orig']:>8.2f} | R²: {row['R2_orig']:>6.3f} | MAPE: {row['MAPE']:>6.2f}%")
    
    return summary_df

# Plot model comparison
def plot_model_comparison(results):

    print("\nCreating model comparison plots:")
    
    # Prepare data for plotting
    models = list(results.keys())
    datasets = ['train', 'test', 'val']
    
    # Extract metrics
    rmse_data = {dataset: [results[model][dataset]['rmse_orig'] for model in models] for dataset in datasets}
    mae_data = {dataset: [results[model][dataset]['mae_orig'] for model in models] for dataset in datasets}
    r2_data = {dataset: [results[model][dataset]['r2_orig'] for model in models] for dataset in datasets}
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    # 1. RMSE Comparison
    x = np.arange(len(models))
    width = 0.25
    
    for i, dataset in enumerate(datasets):
        axes[0, 0].bar(x + i * width, rmse_data[dataset], width, label=dataset.capitalize())
    
    axes[0, 0].set_xlabel('Models')
    axes[0, 0].set_ylabel('RMSE (hours)')
    axes[0, 0].set_title('Root Mean Square Error')
    axes[0, 0].set_xticks(x + width)
    axes[0, 0].set_xticklabels(models, rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. MAE Comparison
    for i, dataset in enumerate(datasets):
        axes[0, 1].bar(x + i * width, mae_data[dataset], width, label=dataset.capitalize())
    
    axes[0, 1].set_xlabel('Models')
    axes[0, 1].set_ylabel('MAE (hours)')
    axes[0, 1].set_title('Mean Absolute Error')
    axes[0, 1].set_xticks(x + width)
    axes[0, 1].set_xticklabels(models, rotation=45)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. R² Comparison
    for i, dataset in enumerate(datasets):
        axes[1, 0].bar(x + i * width, r2_data[dataset], width, label=dataset.capitalize())
    
    axes[1, 0].set_xlabel('Models')
    axes[1, 0].set_ylabel('R² Score')
    axes[1, 0].set_title('R² Score')
    axes[1, 0].set_xticks(x + width)
    axes[1, 0].set_xticklabels(models, rotation=45)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Validation Performance Focus
    val_rmse = [results[model]['val']['rmse_orig'] for model in models]
    val_mae = [results[model]['val']['mae_orig'] for model in models]
    
    axes[1, 1].scatter(val_rmse, val_mae, s=100, alpha=0.7)
    for i, model in enumerate(models):
        axes[1, 1].annotate(model, (val_rmse[i], val_mae[i]), 
                          xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    axes[1, 1].set_xlabel('RMSE (hours)')
    axes[1, 1].set_ylabel('MAE (hours)')
    axes[1, 1].set_title('Validation Set: RMSE vs MAE')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()




# SAVE ACTUAL VS PREDICTED RESULTS (Overall + Category Level)


# Plot feature importance for Tree-based models
def plot_feature_importance(results, feature_columns):
  
    print("Creating feature importance plots:")
    
    # Models with feature importance
    importance_models = ['Random_Forest', 'Gradient_Boosting', 'XGBoost', 'LightGBM']
    
    available_models = [model for model in importance_models if model in results and 'feature_importance' in results[model]]
    
    if not available_models:
        print("No models with feature importance available.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Feature Importance Comparison', fontsize=16, fontweight='bold')
    axes = axes.ravel()
    
    for i, model_name in enumerate(available_models[:4]):
        if i >= 4:
            break
            
        importance = results[model_name]['feature_importance']
        
        # Get top 15 features
        feature_imp_df = pd.DataFrame({
            'feature': feature_columns,
            'importance': importance
        }).sort_values('importance', ascending=False).head(15)
        
        axes[i].barh(range(len(feature_imp_df)), feature_imp_df['importance'])
        axes[i].set_yticks(range(len(feature_imp_df)))
        axes[i].set_yticklabels(feature_imp_df['feature'], fontsize=8)
        axes[i].set_xlabel('Importance')
        axes[i].set_title(f'{model_name} - Top 15 Features')
        axes[i].invert_yaxis()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

# Save model results and metadata
def save_model_results(results, summary_df, feature_columns):

    print("Saving model results:")
    
    import os
    os.makedirs('models', exist_ok=True)
    
    # Save summary
    summary_df.to_csv('models/model_performance_summary.csv', index=False)
    
    # Save detailed results (without model objects and predictions to save space)
    results_for_save = {}
    for model_name, result in results.items():
        results_for_save[model_name] = {
            'train': result['train'],
            'test': result['test'],
            'val': result['val']
        }
        if 'feature_importance' in result:
            results_for_save[model_name]['feature_importance'] = result['feature_importance'].tolist()
    
    with open('models/detailed_results.json', 'w') as f:
        json.dump(results_for_save, f, indent=2)
    
    # Save metadata
    metadata = {
        'experiment_date': datetime.now().isoformat(),
        'models_trained': list(results.keys()),
        'target_variable': 'target_log',
        'num_features': len(feature_columns),
        'feature_columns': feature_columns,
        'best_model_validation': min(results.items(), key=lambda x: x[1]['val']['rmse_orig'])[0]
    }
    
    with open('models/experiment_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("Model results saved successfully!")




# Main modeling pipeline
def main():

    # Create directories
    import os
    os.makedirs('plots', exist_ok=True)
    
    try:
        # Load data
        train_df, test_df, val_df, feature_columns = load_feature_data()
        
        # Prepare data (using log-transformed target)
        X_train, X_test, X_val, y_train, y_test, y_val = prepare_data(
            train_df, test_df, val_df, feature_columns, target='target_log'
        )
        
        # Scale features for neural network
        X_train_scaled, X_test_scaled, X_val_scaled, scaler = scale_features(
            X_train, X_test, X_val, method='standard'
        )
        
        # Train models on original features (tree-based models work better with unscaled features)
        trainer = ModelTrainer(X_train, X_test, X_val, y_train, y_test, y_val, target_type='log')
        trainer.train_all_models()
        
        # # Train neural network on scaled features
        # nn_trainer = ModelTrainer(X_train_scaled, X_test_scaled, X_val_scaled, y_train, y_test, y_val, target_type='log')
        # nn_trainer.train_neural_network()
        
        # Combine results
        all_results = trainer.results
        # all_results['Neural_Network'] = nn_trainer.results['Neural_Network']
        
        # Create summary and visualizations
        summary_df = create_results_summary(all_results)
        plot_model_comparison(all_results)
        plot_feature_importance(all_results, feature_columns)
        
        # Save results
        save_model_results(all_results, summary_df, feature_columns)
        
        print(f"\nModeling pipeline completed successfully!")
        print(f"Results saved in 'models/' directory")
        print(f"Plots saved in 'plots/' directory")
        
    except Exception as e:
        print(f"Error in modeling pipeline: {e}")
        raise



if __name__ == "__main__":
    main()