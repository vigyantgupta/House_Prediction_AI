"""Shared utility functions for model training and prediction."""

import logging
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure structured logging for the application."""
    logger = logging.getLogger("house_predictor")
    if logger.handlers:
        return logger

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def resolve_path(path: str | Path, base_dir: Path) -> Path:
    """Resolve a path relative to base directory."""
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    if path_obj.exists():
        return path_obj.resolve()
    return (base_dir / path_obj).resolve()


@lru_cache(maxsize=1)
def load_model(model_path: Path) -> Any:
    """Load a trained model from joblib file."""
    with open(model_path, "rb") as f:
        return joblib.load(f)


def save_model(model: Any, path: Path) -> None:
    """Save a trained model using joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


@lru_cache(maxsize=1)
def load_preprocessor(preprocessor_path: Path) -> Any:
    """Load the fitted preprocessing pipeline."""
    with open(preprocessor_path, "rb") as f:
        return joblib.load(f)


def save_preprocessor(preprocessor: Any, path: Path) -> None:
    """Save the fitted preprocessing pipeline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)
