"""Data preprocessing and feature engineering pipeline."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .utils import save_preprocessor


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build a preprocessing pipeline with ColumnTransformer.

    Handles numeric imputation + scaling and categorical imputation + encoding.
    """
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore", sparse_output=False, max_categories=50
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features to enrich the dataset.

    Derived features:
    - HouseAge: Current year - YearBuilt
    - RemodAge: Current year - YearRemodAdd
    - HasBasement: 1 if TotalBsmtSF > 0
    - HasGarage: 1 if GarageArea > 0
    - HasPool: 1 if PoolArea > 0
    - TotalArea: GrLivArea + TotalBsmtSF (if present)
    """
    df = df.copy()

    # House age features
    if "YearBuilt" in df.columns:
        df["HouseAge"] = 2024 - df["YearBuilt"]
    if "YearRemodAdd" in df.columns:
        df["RemodAge"] = 2024 - df["YearRemodAdd"]

    # Basement features
    if "TotalBsmtSF" in df.columns:
        df["HasBasement"] = (df["TotalBsmtSF"] > 0).astype(int)

    # Garage features
    if "GarageArea" in df.columns:
        df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
    if "GarageCars" in df.columns and "GarageArea" in df.columns:
        df["GarageQuality"] = df["GarageCars"] * df["GarageArea"]

    # Pool feature
    if "PoolArea" in df.columns:
        df["HasPool"] = (df["PoolArea"] > 0).astype(int)

    # Quality interaction features
    if "OverallQual" in df.columns and "OverallCond" in df.columns:
        df["QualityScore"] = df["OverallQual"] * df["OverallCond"]

    # Area features
    if "GrLivArea" in df.columns and "TotalBsmtSF" in df.columns:
        df["TotalArea"] = df["GrLivArea"] + df["TotalBsmtSF"]

    # Lot features
    if "LotFrontage" in df.columns and "LotArea" in df.columns:
        df["LotShape"] = df["LotArea"] / (df["LotFrontage"] + 1)

    return df


def preprocess_dataset(
    df: pd.DataFrame, fit_preprocessor: bool = False, preprocessor_path: Path | None = None
) -> tuple[pd.DataFrame, ColumnTransformer | None]:
    """
    Preprocess dataset with feature engineering and scaling.

    Args:
        df: Input dataframe
        fit_preprocessor: If True, fit new preprocessor; if False, use existing
        preprocessor_path: Path to save fitted preprocessor

    Returns:
        Preprocessed dataframe and fitted preprocessor
    """
    # Feature engineering
    df_engineered = engineer_features(df)

    # Build and fit preprocessor
    preprocessor = build_preprocessor(df_engineered)
    df_processed = preprocessor.fit_transform(df_engineered)

    # Convert to dataframe
    feature_names = (
        preprocessor.named_transformers_["num"]
        .named_steps["scaler"]
        .get_feature_names_out(
            preprocessor.named_transformers_["num"]
            .named_steps["imputer"]
            .get_feature_names_out()
        )
        .tolist()
    )
    cat_features = (
        preprocessor.named_transformers_["cat"]
        .named_steps["encoder"]
        .get_feature_names_out()
        .tolist()
    )
    all_features = feature_names + cat_features

    df_result = pd.DataFrame(df_processed, columns=all_features)

    # Save preprocessor if requested
    if preprocessor_path:
        save_preprocessor(preprocessor, preprocessor_path)

    return df_result, preprocessor
