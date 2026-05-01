"""FastAPI REST API for house price predictions."""

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.predict import predict_price
from src.utils import load_model

logger = logging.getLogger("house_predictor")
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "best_model.joblib"
METRICS_FILE = BASE_DIR / "model" / "metrics.json"

app = FastAPI(
    title="House Price Prediction API",
    description="ML API for predicting house prices using advanced regression models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (frontend)
frontend_path = BASE_DIR / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")


# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next: Callable) -> Any:
    """Log all HTTP requests with timestamp and duration."""
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} - Status: {response.status_code} - "
        f"Duration: {duration:.3f}s"
    )
    return response


# Pydantic models
class PredictionRequest(BaseModel):
    """Request schema for single house price prediction."""
    features: dict[str, Any] = Field(
        ...,
        example={
            "MSSubClass": 60,
            "MSZoning": "RL",
            "LotFrontage": 65.0,
            "LotArea": 8450,
            "OverallQual": 7,
            "YearBuilt": 2003,
        }
    )


class PredictionResponse(BaseModel):
    """Response schema for prediction endpoint."""
    success: bool
    prediction: float = Field(..., description="Predicted house price in USD")
    confidence: str = Field(
        default="medium",
        description="Confidence level based on model performance"
    )
    model: str = Field(default="best_model", description="Model used for prediction")


class HealthResponse(BaseModel):
    """Response schema for health check."""
    status: str
    message: str
    version: str


class FeaturesResponse(BaseModel):
    """Response schema for features endpoint."""
    num_features: int
    feature_names: list[str]
    feature_types: dict[str, str]


class ModelInfoResponse(BaseModel):
    """Response schema for model info endpoint."""
    model_type: str
    features_count: int
    model_path: str
    preprocessor_path: str
    metrics: dict[str, Any]


class ErrorResponse(BaseModel):
    """Response schema for errors."""
    error: str
    detail: str
    status_code: int


# Endpoints
@app.get("/", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """
    Health check endpoint / Root redirect.

    Returns:
        HealthResponse: Status and version info
    """
    return HealthResponse(
        status="healthy",
        message="House Price Prediction API is running",
        version="1.0.0"
    )


@app.get("/predict-page")
async def predict_page():
    """Redirect to prediction frontend."""
    from fastapi.responses import FileResponse
    frontend_file = BASE_DIR / "frontend" / "index.html"
    if frontend_file.exists():
        return FileResponse(frontend_file)
    return {"error": "Frontend not found"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Predict house price from features.

    Args:
        request: PredictionRequest with feature dictionary

    Returns:
        PredictionResponse with predicted price and confidence

    Raises:
        HTTPException: If prediction fails or model not found
    """
    if not request.features:
        raise HTTPException(
            status_code=400,
            detail="Input data must contain at least one feature."
        )

    try:
        prediction = predict_price(request.features, MODEL_PATH)
        if np.isnan(prediction) or np.isinf(prediction):
            raise ValueError("Prediction resulted in NaN or Inf")

        return PredictionResponse(
            success=True,
            prediction=float(prediction),
            confidence="high",
            model="RandomForestRegressor"
        )
    except FileNotFoundError as exc:
        logger.error(f"Model file not found: {exc}")
        raise HTTPException(
            status_code=503,
            detail="Model file not found. Please train the model before making predictions."
        ) from exc
    except ValueError as exc:
        logger.error(f"Invalid input data: {exc}")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"Prediction error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/features", response_model=FeaturesResponse)
def get_features() -> FeaturesResponse:
    """
    Get expected feature schema for predictions.

    Returns:
        FeaturesResponse with feature names and types

    Raises:
        HTTPException: If model not found
    """
    try:
        model = load_model(MODEL_PATH)
        feature_names = (
            model.feature_names_in_.tolist()
            if hasattr(model, "feature_names_in_")
            else []
        )
        feature_types = {name: "float" for name in feature_names}

        return FeaturesResponse(
            num_features=len(feature_names),
            feature_names=feature_names,
            feature_types=feature_types
        )
    except FileNotFoundError as exc:
        logger.error(f"Model not found: {exc}")
        raise HTTPException(
            status_code=503,
            detail="Model not found. Please train the model first."
        ) from exc


@app.get("/model-info", response_model=ModelInfoResponse)
def get_model_info() -> ModelInfoResponse:
    """
    Get information about the trained model.

    Returns:
        ModelInfoResponse with model metadata

    Raises:
        HTTPException: If model not found
    """
    try:
        model = load_model(MODEL_PATH)
        model_type = type(model).__name__

        metrics = {}
        if METRICS_FILE.exists():
            with open(METRICS_FILE, "r") as f:
                metrics = json.load(f)

        return ModelInfoResponse(
            model_type=model_type,
            features_count=(
                len(model.feature_names_in_)
                if hasattr(model, "feature_names_in_")
                else 0
            ),
            model_path=str(MODEL_PATH),
            preprocessor_path=str(BASE_DIR / "model" / "preprocessor.joblib"),
            metrics=metrics
        )
    except FileNotFoundError as exc:
        logger.error(f"Model not found: {exc}")
        raise HTTPException(
            status_code=503,
            detail="Model not found. Please train the model first."
        ) from exc


@app.get("/metrics")
def get_metrics() -> dict[str, Any]:
    """
    Get model evaluation metrics.

    Returns:
        Dictionary with model performance metrics

    Raises:
        HTTPException: If metrics file not found
    """
    try:
        if not METRICS_FILE.exists():
            return {
                "message": "Metrics not available. Please train the model.",
                "available_metrics": []
            }

        with open(METRICS_FILE, "r") as f:
            metrics = json.load(f)

        return metrics
    except Exception as exc:
        logger.error(f"Error loading metrics: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Error loading metrics"
        ) from exc


@app.get("/docs", include_in_schema=False)
async def custom_docs():
    """API documentation endpoint."""
    from fastapi.openapi.docs import get_swagger_ui_html
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title="House Price Prediction API"
    )


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with custom response."""
    return {
        "error": "HTTPException",
        "detail": exc.detail,
        "status_code": exc.status_code
    }


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return {
        "error": type(exc).__name__,
        "detail": str(exc),
        "status_code": 500
    }
