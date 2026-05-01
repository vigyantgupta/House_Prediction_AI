from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.predict import predict_price


app = FastAPI(title="House Price Prediction API", version="1.0.0")


class PredictionRequest(BaseModel):
    features: dict[str, Any]


class PredictionResponse(BaseModel):
    success: bool
    prediction: float


@app.get("/")
def health_check():
    return {"success": True, "message": "House Price Prediction API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if not request.features:
        raise HTTPException(status_code=400, detail="Input data must contain at least one feature.")

    try:
        prediction = predict_price(request.features)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail="Model file not found. Train the model before calling /predict.",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return PredictionResponse(success=True, prediction=prediction)
