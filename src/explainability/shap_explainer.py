"""SHAP-based model explainability."""
import shap
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path

MODEL_DIR = Path("data/models")


def _load(fn):
    with open(MODEL_DIR / fn, "rb") as f:
        return pickle.load(f)


def get_shap_explainer(model_name: str = "Random Forest"):
    model = _load(f"{model_name.lower().replace(' ','_')}.pkl")
    df = pd.read_csv("data/processed/student_processed.csv")
    feature_cols = _load("feature_cols.pkl")
    imputer = _load("imputer.pkl")
    X = imputer.transform(df[feature_cols].values)

    if model_name == "Logistic Regression":
        scaler = _load("scaler.pkl")
        X = scaler.transform(X)
        explainer = shap.LinearExplainer(model, X)
    else:
        explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X[:200])
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif hasattr(shap_values, 'ndim') and shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    return explainer, shap_values, X[:200], feature_cols


def get_local_shap(student_features: np.ndarray, model_name: str = "Random Forest"):
    """SHAP for a single prediction."""
    model = _load(f"{model_name.lower().replace(' ','_')}.pkl")
    imputer = _load("imputer.pkl")
    X = imputer.transform(student_features.reshape(1, -1))

    if model_name == "Logistic Regression":
        scaler = _load("scaler.pkl")
        X = scaler.transform(X)
        explainer = shap.LinearExplainer(model, X)
    else:
        explainer = shap.TreeExplainer(model)

    vals = explainer.shap_values(X)
    if isinstance(vals, list):
        vals = vals[1]
    elif hasattr(vals, 'ndim') and vals.ndim == 3:
        vals = vals[:, :, 1]
    return vals[0]


def top_features_df(shap_values: np.ndarray, feature_cols: list, n: int = 15) -> pd.DataFrame:
    mean_abs = np.abs(shap_values).mean(axis=0)
    idx = np.argsort(mean_abs)[::-1][:n]
    return pd.DataFrame({
        "Feature": [feature_cols[i] for i in idx],
        "SHAP Importance": mean_abs[idx],
    })
