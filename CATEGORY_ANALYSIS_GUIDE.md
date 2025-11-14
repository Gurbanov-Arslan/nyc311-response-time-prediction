# Model Performance by Category Analysis - Guide

## Overview

This guide explains how to analyze model performance across complaint categories using three different versions.

## Version Options

### Option 1: V3 (Recommended - Simplest) ⭐
**File:** `07_model_performance_by_category_v3.py`

**What it does:**
- Trains lightweight models on training data
- Evaluates predictions on validation and test sets
- **Groups results by complaint category** (THIS IS WHAT YOU WANT!)
- Calculates metrics (RMSE, MAE, R², MAPE) for each category
- Saves results to CSV and Excel with 8 sheets

**How to run:**
```powershell
python 07_model_performance_by_category_v3.py
```

**Output:**
- `models/performance_by_category_v2.csv` - Full results table
- `models/performance_by_category_v2.xlsx` - Excel with 8 sheets:
  1. **Full Results** - All metrics for all categories
  2. **Validation Results** - Validation set only
  3. **Test Results** - Test set only
  4. **Model Summary** - Average metrics by model
  5. **Category Summary** - Average metrics by category
  6. **Pivot R² (Val)** - R² matrix by category and model
  7. **Pivot RMSE (Val)** - RMSE matrix by category and model
  8. **Top & Bottom** - Best and worst performing categories per model

**Plots:**
- `performance_by_category_heatmap_v2.png` - R² heatmap
- `rmse_by_category_top10_v2.png` - RMSE comparison
- `r2_distribution_by_model_v2.png` - R² distribution by model

**Speed:** ~2-3 minutes (trains 3 models on ~40k samples)

---

### Option 2: V2 (Original)
**File:** `07_model_performance_by_category_v2.py`

**What it does:**
- Loads pre-trained models from detailed_results.json
- Extracts predictions by category

**Status:** ⚠️ Requires `y_pred` and `y_true` in detailed_results.json (not currently available)

---

### Option 3: Original
**File:** `07_model_performance_by_category.py`

**What it does:**
- Retrains models from scratch
- Validates the training/prediction pipeline

**Status:** Slower, good for verification

---

## Results Interpretation

### CSV Format
```
dataset,model,category,count,rmse,mae,r2,mape
validation,Random Forest,Noise,235,42.1,28.3,0.65,65.3
validation,Random Forest,Heat,187,38.2,25.1,0.72,58.1
...
```

### Understanding the Metrics

- **count**: Number of samples for this category
- **rmse**: Root Mean Squared Error (hours) - lower is better
- **mae**: Mean Absolute Error (hours) - lower is better
- **r2**: R² Score (0-1) - higher is better, negative means worse than baseline
- **mape**: Mean Absolute Percentage Error (%) - lower is better

### Example Analysis

If you see:
```
Random Forest:
  Best: "Noise" with R²=0.72, RMSE=38.2h
  Worst: "Illegal Parking" with R²=0.42, RMSE=72.1h
```

This means:
- The model predicts "Noise" complaints very well (explains 72% of variance)
- The model struggles with "Illegal Parking" (only 42% of variance)
- You might need to:
  - Investigate why "Illegal Parking" is hard to predict
  - Collect more/better features for that category
  - Consider a separate model for that category

---

## Quick Start

### Step 1: Run the analysis
```powershell
python 07_model_performance_by_category_v3.py
```

### Step 2: Open the results
- View **CSV** in Excel: `models/performance_by_category_v2.csv`
- View **Excel file** with formatted sheets: `models/performance_by_category_v2.xlsx`
- View **visualizations**: Check `plots/` folder

### Step 3: Analyze
1. Open the Excel file `performance_by_category_v2.xlsx`
2. Go to "Model Summary" sheet to see average performance per model
3. Go to "Category Summary" sheet to see which categories are hard/easy to predict
4. Go to "Pivot R² (Val)" to see the heatmap in table form
5. Go to "Top & Bottom" to see best/worst performing categories

---

## Troubleshooting

**Q: Script not running?**
A: Make sure you have:
```powershell
pip install pandas numpy matplotlib seaborn scikit-learn
pip install openpyxl  # Optional but recommended for Excel export
```

**Q: No results in CSV?**
A: Check that:
- `features/train_features.csv`, `features/val_features.csv`, `features/test_features.csv` exist
- `data/train_data.csv`, `data/val_data.csv`, `data/test_data.csv` exist with `complaint_type` column

**Q: Excel file won't open?**
A: Make sure openpyxl is installed: `pip install openpyxl`

---

## File Comparison

| Feature | V3 | V2 | Original |
|---------|----|----|----------|
| **Groups by category** | ✅ | ✅ | ✅ |
| **Calculates metrics** | ✅ | ✅ | ✅ |
| **Excel export** | ✅ | ✅ | ❌ |
| **Fast** | ✅ | N/A | ❌ |
| **Uses existing models** | ❌ | ✅ | ❌ |
| **Retrains models** | ✅ | ❌ | ✅ |

---

## Recommended Workflow

1. Run **V3** to get category performance analysis
2. Review results in Excel file
3. Identify hard-to-predict categories
4. Investigate root causes:
   - Check complaint type description
   - Look for missing/inconsistent data
   - Analyze feature distributions
5. Consider category-specific models or feature engineering

