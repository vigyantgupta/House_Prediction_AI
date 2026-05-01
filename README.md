<div align="center">

# House Prediction AI

An end-to-end machine learning project that predicts residential house prices using feature engineering, model comparison, FastAPI, and a simple web interface.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-Portfolio%20Project-brightgreen?style=for-the-badge)

</div>

---

## Overview

House Prediction AI is a complete machine learning workflow built around the Ames housing dataset. It loads raw housing data, engineers useful features, trains multiple regression models, selects the best performer, and exposes predictions through a FastAPI backend and browser-based frontend.

This project is designed to demonstrate practical ML engineering skills: data preprocessing, model training, API design, testing, documentation, and safe public repository hygiene.

## What This Project Includes

| Area | Details |
| --- | --- |
| Machine Learning | Linear Regression, Ridge, Random Forest, Gradient Boosting |
| Feature Engineering | House age, remodel age, basement/garage/pool flags, total area, quality score |
| Model Selection | Cross-validation, hyperparameter tuning, final model comparison |
| Backend | FastAPI prediction API with health, prediction, feature, metrics, and model-info endpoints |
| Frontend | Simple browser UI for entering house details and viewing predictions |
| Testing | Pytest tests for preprocessing and API behavior |
| Repo Safety | `.env`, virtual environment, logs, cache files, and model artifacts are ignored |

## Project Structure

```text
House_Prediction_AI/
|-- api/
|   `-- main.py                 # FastAPI application
|-- app/
|   `-- main.py                 # Minimal API entry point
|-- data/
|   `-- raw/
|       `-- train.csv           # Training dataset
|-- frontend/
|   `-- index.html              # Browser UI
|-- notebooks/
|   `-- eda.ipynb               # Exploratory analysis notebook
|-- src/
|   |-- preprocess.py           # Feature engineering and preprocessing
|   |-- predict.py              # Prediction logic
|   |-- train.py                # Training and model comparison
|   `-- utils.py                # Shared utilities
|-- tests/
|   |-- test_api.py             # API tests
|   `-- test_preprocess.py      # Preprocessing tests
|-- .env.example                # Safe environment template
|-- requirements.txt            # Python dependencies
|-- run.py                      # Auto-start script
|-- run.bat                     # Windows launcher
`-- README.md
```

Note: trained model files are generated locally inside `model/` and are intentionally ignored for public GitHub safety.

## Quick Start

### 1. Clone The Repository

```bash
git clone https://github.com/vigyantgupta/House_Prediction_AI.git
cd House_Prediction_AI
```

### 2. Create A Virtual Environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train The Model

```bash
python -m src.train
```

This creates the local model artifacts required for prediction:

```text
model/best_model.joblib
model/preprocessor.joblib
```

### 5. Start The API

```bash
uvicorn api.main:app --reload
```

Open these URLs:

```text
API root:      http://localhost:8000
API docs:      http://localhost:8000/docs
Frontend:      http://localhost:8000/predict-page
Static UI:     http://localhost:8000/static/index.html
```

## One-Command Start

On Windows, you can run:

```bash
run.bat
```

Or use Python directly:

```bash
python run.py
```

The launcher creates a virtual environment if needed, installs dependencies, trains the model if it is missing, and starts the API server.

## API Endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/` | Health check |
| POST | `/predict` | Predict house price |
| GET | `/features` | List expected model features |
| GET | `/model-info` | Show model metadata |
| GET | `/metrics` | Return training metrics if available |
| GET | `/docs` | Interactive Swagger docs |

### Example Prediction Request

```json
{
  "features": {
    "MSSubClass": 60,
    "MSZoning": "RL",
    "LotFrontage": 65,
    "LotArea": 8450,
    "OverallQual": 7,
    "OverallCond": 5,
    "YearBuilt": 2003,
    "YearRemodAdd": 2003,
    "GrLivArea": 1710,
    "TotalBsmtSF": 856,
    "GarageCars": 2,
    "GarageArea": 548
  }
}
```

### Example Response

```json
{
  "success": true,
  "prediction": 208736.42,
  "confidence": "high",
  "model": "RandomForestRegressor"
}
```

## Machine Learning Pipeline

The training flow is implemented in `src/train.py`.

```text
Raw data
  -> Feature engineering
  -> Missing value handling
  -> Scaling and encoding
  -> Baseline model training
  -> Cross-validation
  -> Hyperparameter tuning
  -> Final model selection
  -> Local model export
```

Models compared:

| Model | Purpose |
| --- | --- |
| Linear Regression | Simple baseline |
| Ridge Regression | Regularized linear model |
| Random Forest Regressor | Nonlinear ensemble model |
| Gradient Boosting Regressor | Boosted tree-based model |

## Testing

Run all tests:

```bash
pytest tests/ -v
```

Run tests with coverage:

```bash
pytest tests/ -v --cov=src
```

## Environment Configuration

Copy the example file if you want local overrides:

```bash
copy .env.example .env
```

On macOS/Linux:

```bash
cp .env.example .env
```

`.env.example` is safe to commit. Real `.env` files are ignored by Git.

## Public Repository Safety

This repository is prepared for public GitHub sharing:

| File or Folder | Status |
| --- | --- |
| `.env` | Ignored |
| `.venv/` | Ignored |
| `logs/` | Ignored |
| `__pycache__/` | Ignored |
| `model/*.pkl` | Ignored |
| `model/*.joblib` | Ignored |
| `.env.example` | Safe to share |
| `data/raw/train.csv` | Included for reproducible training |

## Tech Stack

| Category | Tools |
| --- | --- |
| Language | Python |
| API | FastAPI, Uvicorn |
| ML | scikit-learn, NumPy, Pandas, Joblib |
| Visualization | Matplotlib |
| Frontend | HTML, CSS, JavaScript |
| Testing | Pytest, HTTPX |

## Future Improvements

- Add GitHub Actions for automated tests
- Add dependency scanning
- Improve API hardening for production deployment
- Add Docker support
- Add model performance charts to the frontend
- Add a deployed demo link

## Author

Built by [Vigyant Gupta](https://github.com/vigyantgupta) as a machine learning portfolio project.

---

<div align="center">

If you found this project useful, consider starring the repository.

</div>
