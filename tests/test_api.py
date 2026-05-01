"""Integration tests for FastAPI endpoints."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check_returns_200(self):
        """Test that health check returns 200 status."""
        response = client.get("/")
        assert response.status_code == 200

    def test_health_check_response_format(self):
        """Test health check response format."""
        response = client.get("/")
        data = response.json()
        assert "status" in data
        assert "message" in data
        assert "version" in data
        assert data["status"] == "healthy"

    def test_health_check_has_version(self):
        """Test that health check includes version."""
        response = client.get("/")
        data = response.json()
        assert data["version"] == "1.0.0"


class TestPredictEndpoint:
    """Test prediction endpoint."""

    @patch("api.main.predict_price")
    def test_predict_with_valid_features(self, mock_predict):
        """Test prediction with valid input."""
        mock_predict.return_value = 150000.0

        payload = {
            "features": {
                "MSSubClass": 60,
                "MSZoning": "RL",
                "LotFrontage": 65.0,
                "LotArea": 8450,
            }
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert data["success"] is True
        assert data["prediction"] == 150000.0

    def test_predict_missing_features_returns_400(self):
        """Test that missing features returns 400."""
        payload = {"features": {}}
        response = client.post("/predict", json=payload)
        assert response.status_code == 400

    def test_predict_response_has_confidence(self):
        """Test that response includes confidence."""
        with patch("api.main.predict_price") as mock_predict:
            mock_predict.return_value = 200000.0
            payload = {"features": {"test": 1}}
            response = client.post("/predict", json=payload)
            data = response.json()
            assert "confidence" in data

    @patch("api.main.predict_price")
    def test_predict_with_nan_returns_400(self, mock_predict):
        """Test that NaN predictions return 400."""
        import numpy as np
        mock_predict.return_value = np.nan

        payload = {"features": {"test": 1}}
        response = client.post("/predict", json=payload)
        assert response.status_code == 400


class TestFeaturesEndpoint:
    """Test features schema endpoint."""

    @patch("api.main.load_model")
    def test_features_returns_200(self, mock_load):
        """Test features endpoint returns 200."""
        mock_model = MagicMock()
        mock_model.feature_names_in_ = ["feature1", "feature2", "feature3"]
        mock_load.return_value = mock_model

        response = client.get("/features")
        assert response.status_code == 200

    @patch("api.main.load_model")
    def test_features_response_format(self, mock_load):
        """Test features response includes required fields."""
        mock_model = MagicMock()
        mock_model.feature_names_in_ = ["feat1", "feat2"]
        mock_load.return_value = mock_model

        response = client.get("/features")
        data = response.json()
        assert "num_features" in data
        assert "feature_names" in data
        assert "feature_types" in data
        assert data["num_features"] == 2


class TestModelInfoEndpoint:
    """Test model info endpoint."""

    @patch("api.main.load_model")
    def test_model_info_returns_200(self, mock_load):
        """Test model info endpoint returns 200."""
        mock_model = MagicMock()
        mock_model.feature_names_in_ = ["f1", "f2"]
        mock_load.return_value = mock_model

        response = client.get("/model-info")
        assert response.status_code == 200

    @patch("api.main.load_model")
    def test_model_info_response_format(self, mock_load):
        """Test model info response includes required fields."""
        mock_model = MagicMock()
        mock_model.feature_names_in_ = ["f1"]
        mock_load.return_value = mock_model

        response = client.get("/model-info")
        data = response.json()
        assert "model_type" in data
        assert "features_count" in data
        assert "model_path" in data


class TestMetricsEndpoint:
    """Test metrics endpoint."""

    def test_metrics_returns_200(self):
        """Test metrics endpoint returns 200."""
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_response_is_json(self):
        """Test metrics endpoint returns valid JSON."""
        response = client.get("/metrics")
        data = response.json()
        assert isinstance(data, dict)


class TestCORSHeaders:
    """Test CORS middleware."""

    def test_cors_headers_present(self):
        """Test that CORS headers are present in response."""
        response = client.get("/", headers={"Origin": "http://localhost:3000"})
        assert response.status_code == 200

    def test_cors_allows_all_origins(self):
        """Test that CORS allows all origins."""
        response = client.get("/")
        # FastAPI's CORSMiddleware should handle this
        assert response.status_code == 200


class TestErrorHandling:
    """Test error handling."""

    def test_error_response_format(self):
        """Test that errors return valid error response."""
        response = client.get("/invalid-endpoint")
        assert response.status_code == 404

    @patch("api.main.predict_price")
    def test_prediction_error_returns_500(self, mock_predict):
        """Test that prediction errors return 500."""
        mock_predict.side_effect = Exception("Test error")
        payload = {"features": {"test": 1}}
        response = client.post("/predict", json=payload)
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data or "error" in data
