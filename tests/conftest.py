"""Shared test utilities and fixtures."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def sample_dataset():
    """Create a small sample dataset for testing."""
    data = {
        "Id": [1, 2, 3, 4, 5],
        "MSSubClass": [60, 20, 60, 70, 60],
        "MSZoning": ["RL", "RL", "RL", "RL", "RL"],
        "LotFrontage": [65.0, 80.0, 68.0, None, 84.0],
        "LotArea": [8450, 9600, 11250, 9550, 14260],
        "OverallQual": [7, 6, 7, 7, 8],
        "OverallCond": [5, 8, 5, 5, 5],
        "YearBuilt": [2003, 1976, 2001, 1915, 2000],
        "YearRemodAdd": [2003, 1976, 2002, 1970, 2000],
        "GrLivArea": [1710, 1262, 1786, 1717, 2198],
        "TotalBsmtSF": [856, 1262, 920, 1188, 1552],
        "GarageArea": [548, 460, 608, 480, 636],
        "GarageCars": [2, 2, 2, 2, 3],
        "SalePrice": [208500, 181500, 223500, 140000, 250000],
    }
    return pd.DataFrame(data)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def model_dir(temp_dir):
    """Create model directory structure."""
    model_dir = temp_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir
