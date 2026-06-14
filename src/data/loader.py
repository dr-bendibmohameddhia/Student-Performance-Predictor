"""Data loading and validation module."""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


class DataLoader:
    """Handles all data ingestion and initial validation."""

    RAW_PATH = Path("data/raw/student_performance.csv")
    PROCESSED_PATH = Path("data/processed/student_processed.csv")

    def load_raw(self) -> pd.DataFrame:
        return pd.read_csv(self.RAW_PATH)

    def load_processed(self) -> pd.DataFrame:
        return pd.read_csv(self.PROCESSED_PATH)

    def get_summary(self, df: pd.DataFrame) -> dict:
        return {
            "n_students": len(df),
            "n_features": df.shape[1],
            "pass_rate": float(df["passed"].mean()),
            "avg_final_grade": float(df["final_grade"].mean()),
            "missing_pct": float(df.isnull().mean().mean()),
            "at_risk_count": int((df["passed"] == 0).sum()),
        }
