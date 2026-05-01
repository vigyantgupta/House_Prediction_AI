"""Model training with multi-model comparison and hyperparameter tuning."""

from io import StringIO
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from .preprocess import preprocess_dataset
from .utils import save_model, save_preprocessor, setup_logging

logger = setup_logging()

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = BASE_DIR / "data" / "raw" / "train.csv"
MODEL_DIR = BASE_DIR / "model"
TARGET_COLUMN = "SalePrice"


def resolve_path(path: str | Path) -> Path:
    """Resolve a path relative to base directory."""
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    if path_obj.exists():
        return path_obj.resolve()
    return (BASE_DIR / path_obj).resolve()


def normalize_dataset_text(raw_text: str) -> str:
    """Normalize dataset text to ensure consistent tab separation."""
    first_line = raw_text.splitlines()[0] if raw_text else ""
    has_literal_tab = "\\t" in first_line
    has_real_tab = "\t" in first_line
    has_comma = "," in first_line

    if has_literal_tab and not has_real_tab and not has_comma:
        return raw_text.replace("\\t", "\t")

    return raw_text


def load_dataset(data_path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load and validate dataset."""
    resolved_path = resolve_path(data_path)
    raw_text = resolved_path.read_text(encoding="utf-8")
    normalized_text = normalize_dataset_text(raw_text)

    for sep, read_kwargs in (
        ("\t", {"sep": "\t"}),
        (",", {"sep": ","}),
        ("auto", {"sep": None, "engine": "python"}),
    ):
        df = pd.read_csv(StringIO(normalized_text), **read_kwargs)
        if df.shape[1] > 1 and TARGET_COLUMN in df.columns:
            if normalized_text != raw_text and sep == "\t":
                resolved_path.write_text(normalized_text, encoding="utf-8")
            return df

    raise ValueError(
        f"Could not load '{resolved_path}' with valid columns. "
        f"Expected to find '{TARGET_COLUMN}' in the dataset."
    )


def train_single_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model: Any,
    model_name: str,
    cv: int = 5,
) -> dict[str, Any]:
    """Train a single model and compute cross-validation metrics."""
    scoring = {"r2": "r2", "neg_mae": "neg_mean_absolute_error", "neg_rmse": "neg_root_mean_squared_error"}
    cv_results = cross_validate(
        model, X_train, y_train, cv=cv, scoring=scoring, return_train_score=True
    )

    train_r2 = cv_results["train_r2"].mean()
    train_mae = -cv_results["train_neg_mae"].mean()
    val_r2 = cv_results["test_r2"].mean()
    val_mae = -cv_results["test_neg_mae"].mean()

    return {
        "model_name": model_name,
        "model": model,
        "train_r2": train_r2,
        "train_mae": train_mae,
        "val_r2": val_r2,
        "val_mae": val_mae,
        "train_r2_std": cv_results["train_r2"].std(),
        "val_r2_std": cv_results["test_r2"].std(),
    }


def tune_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model: Any,
    param_grid: dict[str, Any],
    model_name: str,
) -> dict[str, Any]:
    """Tune a model using GridSearchCV."""
    grid_search = GridSearchCV(
        model, param_grid, cv=5, scoring="r2", n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)

    logger.info(f"{model_name} - Best params: {grid_search.best_params_}")
    logger.info(f"{model_name} - Best CV R²: {grid_search.best_score_:.4f}")

    return {
        "model_name": model_name,
        "model": grid_search.best_estimator_,
        "best_params": grid_search.best_params_,
        "best_cv_score": grid_search.best_score_,
    }


def train_all_models(
    X_train: pd.DataFrame, y_train: pd.Series
) -> list[dict[str, Any]]:
    """Train all baseline models and compare."""
    logger.info("=" * 80)
    logger.info("Training baseline models with 5-fold cross-validation...")
    logger.info("=" * 80)

    models_to_train = [
        (LinearRegression(), "Linear Regression"),
        (Ridge(alpha=1.0), "Ridge Regression"),
        (RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1), "Random Forest"),
        (
            GradientBoostingRegressor(
                n_estimators=100, random_state=42, learning_rate=0.1, max_depth=5
            ),
            "Gradient Boosting",
        ),
    ]

    baseline_results = []
    for model, name in models_to_train:
        logger.info(f"\nTraining {name}...")
        result = train_single_model(X_train, y_train, model, name)
        baseline_results.append(result)
        logger.info(
            f"  CV R² (train/val): {result['train_r2']:.4f} / {result['val_r2']:.4f}"
        )
        logger.info(
            f"  CV MAE (train/val): {result['train_mae']:.2f} / {result['val_mae']:.2f}"
        )

    return baseline_results


def tune_top_models(
    X_train: pd.DataFrame, y_train: pd.Series, baseline_results: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Tune the top 2 performing models."""
    logger.info("\n" + "=" * 80)
    logger.info("Hyperparameter tuning for top models...")
    logger.info("=" * 80)

    # Sort by validation R² to get top models
    sorted_results = sorted(baseline_results, key=lambda x: x["val_r2"], reverse=True)
    top_models = sorted_results[:2]

    tuned_results = []

    for result in top_models:
        model_name = result["model_name"]
        logger.info(f"\n🔧 Tuning {model_name}...")

        if model_name == "Random Forest":
            param_grid = {
                "n_estimators": [100, 200, 300],
                "max_depth": [10, 15, 20, None],
                "min_samples_split": [2, 5, 10],
            }
            tuned = tune_model(
                X_train, y_train, RandomForestRegressor(random_state=42, n_jobs=-1),
                param_grid, model_name
            )

        elif model_name == "Gradient Boosting":
            param_grid = {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 5, 7],
            }
            tuned = tune_model(
                X_train, y_train,
                GradientBoostingRegressor(random_state=42),
                param_grid, model_name
            )

        elif model_name == "Ridge Regression":
            param_grid = {"alpha": [0.1, 1.0, 10.0, 100.0]}
            tuned = tune_model(
                X_train, y_train, Ridge(), param_grid, model_name
            )
        else:
            continue

        tuned_results.append(tuned)

    return tuned_results


def evaluate_final_models(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    baseline_results: list[dict[str, Any]],
    tuned_results: list[dict[str, Any]],
) -> pd.DataFrame:
    """Evaluate all models on final train/val/test splits and return metrics table."""
    logger.info("\n" + "=" * 80)
    logger.info("Final model evaluation on train/val/test splits...")
    logger.info("=" * 80)

    all_results = []

    # Evaluate baseline models
    for result in baseline_results:
        model = result["model"]
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)

        metrics = {
            "Model": result["model_name"],
            "Status": "Baseline",
            "Train R²": r2_score(y_train, y_train_pred),
            "Val R²": r2_score(y_val, y_val_pred),
            "Test R²": r2_score(y_test, y_test_pred),
            "Train RMSE": np.sqrt(mean_squared_error(y_train, y_train_pred)),
            "Val RMSE": np.sqrt(mean_squared_error(y_val, y_val_pred)),
            "Test RMSE": np.sqrt(mean_squared_error(y_test, y_test_pred)),
            "Train MAE": mean_absolute_error(y_train, y_train_pred),
            "Val MAE": mean_absolute_error(y_val, y_val_pred),
            "Test MAE": mean_absolute_error(y_test, y_test_pred),
        }
        all_results.append(metrics)

    # Evaluate tuned models
    for result in tuned_results:
        model = result["model"]
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)

        metrics = {
            "Model": result["model_name"],
            "Status": "Tuned",
            "Train R²": r2_score(y_train, y_train_pred),
            "Val R²": r2_score(y_val, y_val_pred),
            "Test R²": r2_score(y_test, y_test_pred),
            "Train RMSE": np.sqrt(mean_squared_error(y_train, y_train_pred)),
            "Val RMSE": np.sqrt(mean_squared_error(y_val, y_val_pred)),
            "Test RMSE": np.sqrt(mean_squared_error(y_test, y_test_pred)),
            "Train MAE": mean_absolute_error(y_train, y_train_pred),
            "Val MAE": mean_absolute_error(y_val, y_val_pred),
            "Test MAE": mean_absolute_error(y_test, y_test_pred),
        }
        all_results.append(metrics)

    metrics_df = pd.DataFrame(all_results)
    return metrics_df


def get_best_model(
    baseline_results: list[dict[str, Any]], tuned_results: list[dict[str, Any]]
) -> tuple[Any, str]:
    """Select best model based on validation R²."""
    all_candidates = [
        (result["model"], result["model_name"], result["val_r2"])
        for result in baseline_results
    ] + [
        (result["model"], result["model_name"], result.get("best_cv_score", -np.inf))
        for result in tuned_results
    ]

    best_model, best_name, best_score = max(all_candidates, key=lambda x: x[2])
    logger.info(f"\n✅ Best model: {best_name} (val R²: {best_score:.4f})")
    return best_model, best_name


def save_feature_importance(model: Any, X_train: pd.DataFrame, model_name: str) -> None:
    """Save feature importance plot for tree-based models."""
    if not hasattr(model, "feature_importances_"):
        logger.info(f"⚠ {model_name} does not have feature_importances_ attribute")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[-15:]

    plt.figure(figsize=(10, 6))
    plt.title(f"{model_name} - Top 15 Feature Importances")
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [X_train.columns[i] for i in indices])
    plt.xlabel("Importance")
    plt.tight_layout()

    importance_path = MODEL_DIR / "feature_importance.png"
    plt.savefig(importance_path, dpi=100, bbox_inches="tight")
    plt.close()
    logger.info(f"💾 Feature importance plot saved to {importance_path}")


def train_model(data_path: str | Path = DEFAULT_DATA_PATH) -> Any:
    """
    Complete training pipeline: load data, preprocess, train multiple models,
    tune best ones, and save the winner.
    """
    logger.info("🚀 Starting model training pipeline...")

    # Load and preprocess data
    logger.info(f"Loading dataset from {resolve_path(data_path)}...")
    df = load_dataset(data_path)
    logger.info(f"Dataset shape: {df.shape}")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    logger.info("Preprocessing data with feature engineering...")
    X_processed, preprocessor = preprocess_dataset(X, preprocessor_path=MODEL_DIR / "preprocessor.joblib")

    # Split data: 70% train, 15% val, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_processed, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    logger.info(f"Data splits: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # Train baseline models
    baseline_results = train_all_models(X_train, y_train)

    # Tune top models
    tuned_results = tune_top_models(X_train, y_train, baseline_results)

    # Evaluate all models
    metrics_df = evaluate_final_models(
        X_train, X_val, X_test, y_train, y_val, y_test,
        baseline_results, tuned_results
    )

    # Print summary table
    logger.info("\n" + "=" * 80)
    logger.info("FINAL MODEL COMPARISON")
    logger.info("=" * 80)
    print(metrics_df.to_string(index=False))
    logger.info("=" * 80)

    # Select and save best model
    best_model, best_name = get_best_model(baseline_results, tuned_results)
    best_model.fit(X_train, y_train)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_model(best_model, MODEL_DIR / "best_model.joblib")
    logger.info(f"💾 Best model saved to {MODEL_DIR / 'best_model.joblib'}")

    # Save feature importance if applicable
    save_feature_importance(best_model, X_processed, best_name)

    logger.info("\n✨ Training complete!")
    return best_model


if __name__ == "__main__":
    train_model()
