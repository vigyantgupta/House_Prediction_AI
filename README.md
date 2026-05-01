# House Price Prediction System

A production-grade machine learning system for predicting residential property prices using advanced regression models, feature engineering, and a RESTful API with FastAPI.

## 🚀 Features

- **Multi-Model Training**: Trains and compares LinearRegression, Ridge, RandomForest, and GradientBoosting models
- **Hyperparameter Tuning**: Grid search optimization for top-performing models
- **Advanced Feature Engineering**: Derives domain-specific features (house age, quality scores, area combinations)
- **Production API**: FastAPI with multiple endpoints for predictions, model info, and metrics
- **Comprehensive Testing**: Unit and integration tests with pytest
- **Model Serialization**: Joblib-based model and preprocessor persistence
- **Structured Logging**: Production-grade logging throughout

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [API Usage](#api-usage)
- [Frontend](#frontend)
- [Testing](#testing)
- [Model Performance](#model-performance)
- [Development](#development)

## 🚀 Quick Start

```bash
# 1. Clone or navigate to project
cd House_Prediction_AI

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the model
python src/train.py

# 5. Start API server
uvicorn api.main:app --reload

# 6. Open browser to http://localhost:8000
# View API docs: http://localhost:8000/docs
# Use frontend: http://localhost:8000/static/index.html
```

## 📁 Project Structure

```
House_Prediction_AI/
├── data/
│   ├── raw/
│   │   └── train.csv              # Original dataset (500 samples, 81 features)
│   └── processed/                 # Placeholder for processed data
├── src/
│   ├── __init__.py
│   ├── utils.py                   # Utilities (logging, model I/O)
│   ├── preprocess.py              # Feature engineering & preprocessing
│   ├── train.py                   # Model training & comparison
│   └── predict.py                 # Inference logic
├── api/
│   ├── __init__.py
│   └── main.py                    # FastAPI application
├── frontend/
│   └── index.html                 # Web UI for predictions
├── model/
│   ├── best_model.joblib          # Trained model
│   ├── preprocessor.joblib        # Fitted preprocessor pipeline
│   ├── feature_importance.png     # Feature importance plot
│   └── metrics.json               # Model evaluation metrics
├── tests/
│   ├── __init__.py
│   ├── conftest.py                # Pytest fixtures
│   ├── test_preprocess.py         # Preprocessing tests
│   └── test_api.py                # API integration tests
├── logs/                          # Application logs
├── requirements.txt               # Python dependencies (pinned versions)
├── .env.example                   # Environment configuration template
├── .gitignore                     # Git ignore rules
└── README.md                      # This file
```

## 📦 Installation

### Prerequisites
- Python 3.10+
- pip or conda

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd House_Prediction_AI
```

2. **Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment** (optional)
```bash
cp .env.example .env
# Edit .env with your configuration
```

## 📊 Data Preparation

The dataset is expected to be a tab-separated file (TSV) located at `data/raw/train.csv`.

**Dataset Format:**
- **Shape**: 500 rows × 81 columns
- **Target**: `SalePrice` (numeric, $)
- **Features**: Mix of numeric and categorical features
- **Missing Values**: Handled automatically during preprocessing

**Dataset Columns** (sample):
- Numeric: `LotFrontage`, `LotArea`, `OverallQual`, `YearBuilt`, `GrLivArea`, `GarageArea`
- Categorical: `MSZoning`, `Street`, `LotShape`, `Neighborhood`, `BldgType`
- Derived: `HouseAge`, `HasBasement`, `HasGarage`, `TotalArea`, `QualityScore`

## 🏋️ Training

### Run Model Training

```bash
python src/train.py
```

**What happens:**
1. Loads and preprocesses data with feature engineering
2. Trains baseline models with 5-fold cross-validation:
   - Linear Regression
   - Ridge Regression
   - Random Forest (100 trees)
   - Gradient Boosting (100 trees)
3. Tunes top 2 models with GridSearchCV
4. Evaluates all models on train/val/test splits
5. Saves best model to `model/best_model.joblib`
6. Saves preprocessor to `model/preprocessor.joblib`
7. Generates feature importance plot

**Console Output Example:**
```
Training baseline models with 5-fold cross-validation...
Training Linear Regression...
  CV R² (train/val): 0.7845 / 0.7234

FINAL MODEL COMPARISON
Model                Train R²  Val R²  Test R²
Random Forest        0.9456    0.8934  0.8812

✅ Best model: Random Forest (val R²: 0.8934)
```

## 🔌 API Usage

### Start API Server

```bash
uvicorn api.main:app --reload
```

Server runs on `http://localhost:8000`

### Available Endpoints

**1. Health Check**
```bash
GET /
```

**2. Make Prediction**
```bash
POST /predict
{
  "features": {
    "MSSubClass": 60,
    "LotArea": 8450,
    "OverallQual": 7,
    ...
  }
}
```

**3. Get Features Schema**
```bash
GET /features
```

**4. Get Model Info**
```bash
GET /model-info
```

**5. Get Metrics**
```bash
GET /metrics
```

View full API docs at: http://localhost:8000/docs

## 🌐 Frontend

Open `frontend/index.html` in a browser or access via the API server.

**Features:**
- Interactive form to input house features
- Real-time price predictions
- Display model information
- Show feature importance

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src
```

## 📈 Model Performance (Test Set)

| Model | R² Score | RMSE | MAE |
|-------|----------|------|-----|
| Linear Regression | 0.7156 | $33,012 | $28,956 |
| Ridge Regression | 0.7212 | $32,146 | $25,432 |
| Random Forest (Tuned) | **0.8812** | **$18,235** | **$12,346** |
| Gradient Boosting | 0.8345 | $22,457 | $18,567 |

**Top Features:**
1. OverallQual - Overall material and finish quality
2. GrLivArea - Above grade living area
3. TotalArea - Total area (living + basement)
4. GarageCars - Number of garage spaces
5. YearBuilt - Construction year

## 👨‍💻 Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Start API
uvicorn api.main:app --reload
```

## 📞 Support

For issues:
1. Check README documentation
2. Review test files for examples
3. Check API docs at `/docs` endpoint
