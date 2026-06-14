"""Model loading and prediction service."""
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple


MODEL_DIR = Path("data/models")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Random Forest":       "random_forest.pkl",
    "XGBoost":             "xgboost.pkl",
    "LightGBM":            "lightgbm.pkl",
}


def load_artifact(filename: str):
    with open(MODEL_DIR / filename, "rb") as f:
        return pickle.load(f)


def load_results() -> dict:
    with open(MODEL_DIR / "results.json") as f:
        return json.load(f)


def get_best_model_name() -> str:
    return load_results()["best_model"]


def predict_single(features: np.ndarray, model_name: str = None) -> Tuple[int, float]:
    """Predict pass/fail and probability for a single student."""
    if model_name is None:
        model_name = get_best_model_name()

    model = load_artifact(MODEL_FILES[model_name])
    imputer = load_artifact("imputer.pkl")

    X = imputer.transform(features.reshape(1, -1))

    if model_name == "Logistic Regression":
        scaler = load_artifact("scaler.pkl")
        X = scaler.transform(X)

    pred  = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0, 1])
    return pred, proba


def predict_batch(df: pd.DataFrame, model_name: str = None) -> np.ndarray:
    """Predict for a batch DataFrame."""
    if model_name is None:
        model_name = get_best_model_name()

    feature_cols = load_artifact("feature_cols.pkl")
    model  = load_artifact(MODEL_FILES[model_name])
    imputer = load_artifact("imputer.pkl")

    X = imputer.transform(df[feature_cols].values)
    if model_name == "Logistic Regression":
        scaler = load_artifact("scaler.pkl")
        X = scaler.transform(X)

    return model.predict_proba(X)[:, 1]
