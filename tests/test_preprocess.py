"""Tests for data preprocessing pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.preprocess import engineer_features, preprocess_dataset


class TestFeatureEngineering:
    """Test feature engineering functions."""

    def test_engineer_features_house_age(self, sample_dataset):
        """Test house age feature engineering."""
        df = engineer_features(sample_dataset)
        assert "HouseAge" in df.columns
        assert df["HouseAge"].iloc[0] > 0
        assert all(df["HouseAge"] >= 0)

    def test_engineer_features_has_basement(self, sample_dataset):
        """Test basement indicator feature."""
        df = engineer_features(sample_dataset)
        assert "HasBasement" in df.columns
        assert df["HasBasement"].isin([0, 1]).all()

    def test_engineer_features_has_garage(self, sample_dataset):
        """Test garage indicator feature."""
        df = engineer_features(sample_dataset)
        assert "HasGarage" in df.columns
        assert df["HasGarage"].isin([0, 1]).all()

    def test_engineer_features_total_area(self, sample_dataset):
        """Test total area feature."""
        df = engineer_features(sample_dataset)
        if "TotalArea" in df.columns:
            assert df["TotalArea"].iloc[0] == df["GrLivArea"].iloc[0] + df["TotalBsmtSF"].iloc[0]

    def test_engineer_features_preserves_original(self, sample_dataset):
        """Test that original columns are preserved."""
        df = engineer_features(sample_dataset)
        assert "LotArea" in df.columns
        assert "GrLivArea" in df.columns


class TestPreprocessing:
    """Test preprocessing pipeline."""

    def test_preprocess_returns_dataframe(self, sample_dataset):
        """Test that preprocessing returns a DataFrame."""
        X_processed, preprocessor = preprocess_dataset(sample_dataset.drop("SalePrice", axis=1))
        assert isinstance(X_processed, pd.DataFrame)

    def test_preprocess_returns_preprocessor(self, sample_dataset):
        """Test that preprocessing returns a preprocessor object."""
        X_processed, preprocessor = preprocess_dataset(sample_dataset.drop("SalePrice", axis=1))
        assert preprocessor is not None
        assert hasattr(preprocessor, "transform")

    def test_preprocess_handles_missing_values(self):
        """Test that preprocessing handles missing values."""
        df = pd.DataFrame({
            "numeric_col": [1.0, 2.0, np.nan, 4.0, 5.0],
            "categorical_col": ["A", "B", None, "A", "B"],
        })
        X_processed, _ = preprocess_dataset(df)
        assert not X_processed.isnull().any().any()

    def test_preprocess_output_shape(self, sample_dataset):
        """Test that preprocessed output has expected shape."""
        X = sample_dataset.drop("SalePrice", axis=1)
        X_processed, _ = preprocess_dataset(X)
        assert X_processed.shape[0] == len(X)
        assert X_processed.shape[1] > 0

    def test_preprocess_numeric_scaling(self, sample_dataset):
        """Test that numeric features are scaled."""
        X = sample_dataset.drop("SalePrice", axis=1)
        X_processed, _ = preprocess_dataset(X)
        # Check that some features are scaled to [-3, 3] range (roughly)
        numeric_cols = [col for col in X_processed.columns if "num" in col.lower()]
        if numeric_cols:
            sample_values = X_processed[numeric_cols[0]].dropna()
            assert abs(sample_values.max()) < 10 or abs(sample_values.min()) < 10

    def test_preprocess_consistent_output(self, sample_dataset):
        """Test that preprocessing gives consistent results."""
        X = sample_dataset.drop("SalePrice", axis=1)
        X_processed1, _ = preprocess_dataset(X)
        X_processed2, _ = preprocess_dataset(X)
        pd.testing.assert_frame_equal(X_processed1, X_processed2)


class TestPreprocessingSaving:
    """Test preprocessing pipeline saving functionality."""

    def test_save_preprocessor(self, sample_dataset, model_dir):
        """Test saving preprocessor to file."""
        X = sample_dataset.drop("SalePrice", axis=1)
        preprocessor_path = model_dir / "test_preprocessor.joblib"
        X_processed, preprocessor = preprocess_dataset(
            X, preprocessor_path=preprocessor_path
        )
        assert preprocessor_path.exists()
        assert preprocessor_path.stat().st_size > 0
