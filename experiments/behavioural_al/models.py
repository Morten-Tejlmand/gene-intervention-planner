# models for the AL experiments
# TODO: could try MC-dropout on a neural net instead

import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.preprocessing import StandardScaler


class GBTClassifier:
    """Single GBT classifier. Uncertainty = Bernoulli std sqrt(p*(1-p))."""

    def __init__(self, n_estimators=100, random_state=42):
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=3,
            learning_rate=0.1,
            random_state=random_state,
        )
        self.scaler = StandardScaler()

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        proba = self.model.predict_proba(X_scaled)[:, 1]
        return proba, np.sqrt(proba * (1.0 - proba))


class EnsembleClassifier:
    """Ensemble of RF + GBT + ExtraTrees. Std across predictions = uncertainty signal."""

    def __init__(self, n_estimators=100, random_state=42):
        self.models = [
            RandomForestClassifier(
                n_estimators=n_estimators,
                class_weight="balanced",
                random_state=random_state,
            ),
            GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=3,
                learning_rate=0.1,
                random_state=random_state,
            ),
            ExtraTreesClassifier(
                n_estimators=n_estimators,
                class_weight="balanced",
                random_state=random_state,
            ),
        ]
        self.scaler = StandardScaler()

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        for m in self.models:
            m.fit(X_scaled, y)
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        all_probas = np.stack([m.predict_proba(X_scaled)[:, 1] for m in self.models])
        return all_probas.mean(axis=0), all_probas.std(axis=0)


class EnsembleRegressor:
    """Regression version of the ensemble. Not used in current experiments."""
    # TODO: might be useful for continuous phenotype prediction later

    def __init__(self, n_estimators=100, random_state=42):
        self.models = [
            RandomForestRegressor(n_estimators=n_estimators, random_state=random_state),
            GradientBoostingRegressor(
                n_estimators=n_estimators, max_depth=3,
                learning_rate=0.1, random_state=random_state,
            ),
            ExtraTreesRegressor(n_estimators=n_estimators, random_state=random_state),
        ]
        self.scaler = StandardScaler()

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        for m in self.models:
            m.fit(X_scaled, y)
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        preds = np.stack([m.predict(X_scaled) for m in self.models])
        return preds.mean(axis=0), preds.std(axis=0)
