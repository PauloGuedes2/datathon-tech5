from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

from src.config.settings import Settings


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoders = {}

    def fit(self, X, y=None):
        for col in Settings.FEATURES_CATEGORICAS:
            if col in X.columns:
                le = LabelEncoder()
                le.fit(X[col].astype(str))
                self.encoders[col] = le
        return self

    def transform(self, X):
        X_out = X.copy()
        for col, le in self.encoders.items():
            if col in X_out.columns:
                # Trata valores não vistos como -1
                X_out[col] = X_out[col].astype(str).map(
                    lambda s: le.transform([s])[0] if s in le.classes_ else -1
                )
        return X_out
