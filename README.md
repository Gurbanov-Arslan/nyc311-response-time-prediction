# NYC 311 Service Request Resolution Time Prediction

A comprehensive machine learning project to predict resolution times for NYC 311 service requests using multiple algorithms and extensive feature engineering.

## 🎯 Project Objective

Predict the resolution time for NYC 311 service requests to help city agencies:
- Better allocate resources
- Set realistic expectations for citizens
- Improve overall service delivery efficiency

## 📊 Results Summary

### Best Model Performance
- **Random Forest**: Random Forest achieved the best performances on validation set (RMSE: 15.23h, MAE: 8.45h, R²: 0.72)
- **Tree-based models** significantly outperformed linear approaches
- **79 engineered features** capturing temporal, geographic, and complaint-specific patterns

### Key Insights
- Resolution times are highly skewed 
- Complaint type is the strongest predictor
- Geographic location (borough) shows significant impact
- Temporal patterns exist (time of day, day of week effects)

## 🗂️ Project Structure

```
ny-311-service-requests/
├── 01_data_download.py                      # Download data from NYC Open Data API
├── 02_data_preprocessing.py                 # Data cleaning and train/test splits
├── 03_exploratory_analysis.py               # Comprehensive EDA with visualizations
├── 04_feature_engineering.py                # Feature creation and encoding
├── 05_machine_learning_models.py            # Model training and evaluation
├── 06_project_summary.py                    # Final project summary
├── 07_model_performance_by_category.py      # Performance analysis by complaint type
├── 08_model_performance_by_time_bins.py     # Performance analysis by time bins
├── run_pipeline.py                          # Complete pipeline runner
├── setup.sh                                 # Environment setup script (Unix/macOS)
├── setup.bat                                # Environment setup script (Windows)
├── requirements.txt                         # Python package dependencies
├── README.md                                # Project documentation
├── data/                                    # Raw and preprocessed datasets
├── features/                                # Feature-engineered datasets
├── models/                                  # Model results and performance metrics
└── plots/                                   # Visualizations and analysis charts
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Installation

#### Option 1: Automated Setup (Recommended)
```bash
# Clone the repository
git clone <repository-url>
cd ny-311-service-requests

# Run setup script (Unix/macOS)
./setup.sh

# Or for Windows
setup.bat
```

#### Option 2: Manual Setup
```bash
# Clone the repository
git clone <repository-url>
cd ny-311-service-requests

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline

#### Option 1: Complete Pipeline (Recommended)
```bash
# Activate environment (if not already active)
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Run complete pipeline
python run_pipeline.py
```

#### Option 2: Step-by-Step Execution
```bash
# Activate environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Run each step individually
python 01_data_download.py
python 02_data_preprocessing.py
python 03_exploratory_analysis.py
python 04_feature_engineering.py
python 05_machine_learning_models.py
python 06_project_summary.py
python 07_model_performance_by_category.py
python 08_model_performance_by_time_bins.py
```

## 📈 Dataset Information

- **Source**: NYC Open Data API (311 Service Requests)
- **Time Period**: November 2023
- **Split**: 70% train, 15% test, 15% validation
- **Features**: 79 engineered features

## 🔧 Feature Engineering

### Temporal Features (29)
- Hour, day, month components
- Business hours indicators
- Weekend/weekday flags
- Cyclical encoding (sin/cos transformations)

### Complaint Features (22)
- Complaint type frequency
- Average resolution time by type
- Urgency categorization
- Top complaint type indicators

### Geographic Features (23)
- Borough indicators
- ZIP code features
- Location type encoding
- Geographic frequency patterns

### Agency Features (15)
- Responsible agency indicators
- Agency workload patterns
- Average resolution times by agency

### Interaction Features (5)
- Cross-feature relationships
- Frequency × temporal interactions

## 🤖 Models Implemented
## Models performances on unseen dataset (validation dataset)

|Model	          |RMSE   | MAE   |  R2	 |     
|-----------------|---------------|------|
|Random_Forest    |	41.12 | 17.26 |	0.74 |
|Gradient_Boosting| 41.68 | 17.37 |	0.75 |
|XGBoost          |	40.89 | 17.05 |	0.76 |
|LightGBM         |	41.84 | 17.48 |	0.75 |


## 📊 Key Findings

### Data Patterns
- **Quick Resolution**: Parking violations, noise complaints (< 3 hours)
- **Long Resolution**: Housing issues, sanitation problems (> 500 hours)
- **Peak Times**: More requests during business hours and weekdays
- **Borough Differences**: BRONX has higher average resolution times while QUEENS has the lowest

### Business Insights
- Complaint type average resolution time, agency type and descriptor lenght are the strongest predictors
- Geographic location adds significant predictive power
- Temporal patterns help optimize resource allocation
- Agency workload affects resolution efficiency

## 🔮 Future Improvements

### Model Enhancement
- Hyperparameter tuning with GridSearch/RandomSearch
- Ensemble methods combining top performers
- Deep learning with LSTM for temporal sequences
- Real-time model updates with streaming data

### Data Enrichment
- Weather data integration
- City events and holidays
- Traffic and transportation data
- Citizen satisfaction scores

### Deployment
- Real-time prediction API
- Dashboard for city agencies
- Mobile app integration
- Performance monitoring system

## 📊 Visualizations

The project generates several visualization files in the `plots/` directory:
- `resolution_time_analysis.png` - Distribution and patterns analysis
- `complaint_type_analysis.png` - Complaint type insights
- `temporal_analysis.png` - Time-based patterns
- `geographic_analysis.png` - Borough-level analysis
- `correlation_analysis.png` - Feature correlation heatmap
- `model_comparison.png` - Model performance comparison
- `feature_importance.png` - Top predictive features

## 🙏 Acknowledgments

- NYC Open Data for providing the 311 service request dataset
- Scikit-learn, XGBoost, and LightGBM communities for excellent ML libraries
- NYC Department of Information Technology & Telecommunications for maintaining the 311 system

---

**Project Status**: ✅ Complete  
**Last Updated**: November 2025  
**Author**: Arslan Gurbanov