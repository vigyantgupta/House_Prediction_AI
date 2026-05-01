"""Prediction logic for house price estimation."""

from pathlib import Path
from typing import Any

import pandas as pd

from .preprocess import engineer_features
from .utils import load_model, load_preprocessor

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "best_model.joblib"
PREPROCESSOR_PATH = BASE_DIR / "model" / "preprocessor.joblib"


def predict_price(input_data: dict[str, Any], model_path: Path = MODEL_PATH) -> float:
    """
    Predict house price from raw features.

    Args:
        input_data: Dictionary of raw feature values
        model_path: Path to trained model

    Returns:
        Predicted price
    """
    if not input_data:
        raise ValueError("Input data must contain at least one feature.")

    # Load model and preprocessor
    model = load_model(model_path)
    preprocessor_path = model_path.parent / "preprocessor.joblib"

    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}")

    preprocessor = load_preprocessor(preprocessor_path)

    # Convert input to DataFrame and apply feature engineering
    input_df = pd.DataFrame([input_data])
    input_engineered = engineer_features(input_df)

    # Get expected columns from preprocessor (same order as training)
    numeric_features = preprocessor.transformers_[0][2]
    categorical_features = preprocessor.transformers_[1][2]
    all_features = numeric_features + categorical_features

    # Reindex to match training features (fill missing with default values)
    input_engineered = input_engineered.reindex(columns=all_features, fill_value=0)

    # Apply preprocessing
    input_processed = preprocessor.transform(input_engineered)

    # Make prediction
    prediction = model.predict(input_processed)[0]
    return float(prediction)
