"""Feature engineering and preprocessing pipeline."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer


class FeatureEngineer:
    """Transforms raw data into ML-ready features."""

    CAT_COLS = ['gender','address','family_size','parent_status',
                'mother_job','father_job','reason','guardian']

    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Impute
        for col in ['mother_education','father_education','health_status']:
            if col in df.columns:
                df[col].fillna(df[col].median(), inplace=True)
        # Encode
        for col in self.CAT_COLS:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        # Engineer
        df = self._engineer(df)
        return df

    def _engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        df['parent_edu_avg']    = (df['mother_education'] + df['father_education']) / 2
        df['alcohol_total']     = df['workday_alcohol'] + df['weekend_alcohol']
        df['support_score']     = df['school_support'] + df['family_support']
        df['academic_history']  = (df['grade_period1'] + df['grade_period2']) / 2
        df['grade_improvement'] = df['grade_period2'] - df['grade_period1']
        df['study_efficiency']  = df['study_time'] / (df['go_out'] + 1)
        df['at_risk_score']     = df['past_failures'] * 2 + df['alcohol_total'] + df['absences'] // 5
        df['absence_category']  = pd.cut(df['absences'], bins=[-1,2,7,15,50],
                                         labels=[0,1,2,3]).astype(int)
        return df

    def get_feature_names(self, df: pd.DataFrame) -> list:
        return [c for c in df.columns if c not in ['passed','final_grade']]
