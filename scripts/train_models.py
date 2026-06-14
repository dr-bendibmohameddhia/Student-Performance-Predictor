#!/usr/bin/env python3
"""
Train all models and save artifacts.
Run: python scripts/train_models.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import pickle, json, warnings
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import xgboost as xgb
import lightgbm as lgb
warnings.filterwarnings("ignore")

MODEL_DIR = Path("data/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("📂 Loading processed data...")
df = pd.read_csv("data/processed/student_processed.csv")

feature_cols = [c for c in df.columns if c not in ["passed", "final_grade"]]
X = df[feature_cols].values
y = df["passed"].values

# Impute
imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

MODELS = {
    "Logistic Regression": (LogisticRegression(max_iter=1000, C=1.0, random_state=42), True),
    "Random Forest":       (RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5, random_state=42), False),
    "XGBoost":             (xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8, random_state=42, eval_metric="logloss", verbosity=0), False),
    "LightGBM":            (lgb.LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1), False),
}

results = {}
print("\n🚀 Training models...\n")
print(f"{'Model':<22} {'Accuracy':>10} {'AUC':>10} {'F1':>10}")
print("-" * 56)

for name, (model, scaled) in MODELS.items():
    X_tr = X_train_sc if scaled else X_train
    X_te = X_test_sc  if scaled else X_test
    model.fit(X_tr, y_train)
    preds  = model.predict(X_te)
    probas = model.predict_proba(X_te)[:, 1]
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probas)
    f1  = f1_score(y_test, preds)
    results[name] = {"accuracy": round(acc, 4), "auc": round(auc, 4), "f1": round(f1, 4)}
    print(f"{name:<22} {acc:>10.4f} {auc:>10.4f} {f1:>10.4f}")
    # Save
    fname = name.lower().replace(" ", "_")
    with open(MODEL_DIR / f"{fname}.pkl", "wb") as f:
        pickle.dump(model, f)

best = max(results, key=lambda k: results[k]["auc"])

with open(MODEL_DIR / "scaler.pkl",  "wb") as f: pickle.dump(scaler, f)
with open(MODEL_DIR / "imputer.pkl", "wb") as f: pickle.dump(imputer, f)
with open(MODEL_DIR / "feature_cols.pkl", "wb") as f: pickle.dump(feature_cols, f)
with open(MODEL_DIR / "results.json", "w") as f:
    json.dump({"results": results, "best_model": best, "feature_cols": feature_cols}, f, indent=2)

print(f"\n✅ Best model: {best} (AUC: {results[best]['auc']:.4f})")
print("✅ All artifacts saved to data/models/")
