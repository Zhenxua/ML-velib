"""The two references required by the brief and the model families seen in the course.

Every ML model is a scikit-learn estimator/Pipeline that is only ever `fit` on the training rows,
so imputers and scalers inside the pipelines learn their statistics from the training period only.
"""
from dataclasses import dataclass
from functools import partial
from typing import Callable

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .config import EMPTY_THRESHOLD, SEED
from .features import FEATURES, FEATURES_NO_DOW


class Persistence(ClassifierMixin, BaseEstimator):
    """Temporal reference: the state at t + h equals the state at t.

    predict       -> 1{bikes_available(t) < 2}: the fixed rule, no threshold tuned on data.
    predict_proba -> 1 / (1 + bikes_available(t)), a score that decreases with the number of bikes.
                     It only ranks samples (PR curve, PR-AUC); it is not a calibrated probability.
                     score >= 0.5  <=>  bikes <= 1, so it agrees with predict.
    Expects a single input column: bikes_available.
    """

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        return (np.asarray(X, dtype=float)[:, 0] < EMPTY_THRESHOLD).astype(int)

    def predict_proba(self, X):
        p = 1.0 / (1.0 + np.clip(np.asarray(X, dtype=float)[:, 0], 0, None))
        return np.column_stack([1 - p, p])


class FeatureWeighter(TransformerMixin, BaseEstimator):
    """Multiply (standardised) columns by fixed weights -> weighted Euclidean distance for k-NN.
    A named class rather than FunctionTransformer(lambda ...): picklable and the weights are explicit."""

    def __init__(self, weights=None):
        self.weights = weights

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float) * np.asarray(self.weights, dtype=float)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str                 # "reference" or "ml"
    make: Callable[[], BaseEstimator]
    features: tuple
    tune_threshold: bool        # ML: F1-optimal threshold chosen on validation; references: fixed rule


def naive_bayes():
    return make_pipeline(SimpleImputer(strategy="median"), GaussianNB())


def knn(k: int, weights=None):
    steps = [SimpleImputer(strategy="median"), StandardScaler()]
    if weights is not None:
        steps.append(FeatureWeighter(weights))
    return make_pipeline(*steps, KNeighborsClassifier(n_neighbors=k, n_jobs=-1))


def tree(depth: int):
    return make_pipeline(SimpleImputer(strategy="median"),
                         DecisionTreeClassifier(max_depth=depth, min_samples_leaf=50, random_state=SEED))


def forest(seed: int = SEED):
    return make_pipeline(SimpleImputer(strategy="median"),
                         RandomForestClassifier(n_estimators=200, min_samples_leaf=20, max_features="sqrt",
                                                n_jobs=-1, random_state=seed))


def boosting(seed: int = SEED):
    # early_stopping=False on purpose: sklearn's built-in early stopping holds out a *random*
    # fraction of the training rows. NaN (missing lags) are handled natively, no imputer needed.
    # With more than 200 000 training rows the bin edges come from a random subsample, so the seed matters.
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, early_stopping=False, random_state=seed)


def references() -> list[ModelSpec]:
    return [
        ModelSpec("majority class", "reference", partial(DummyClassifier, strategy="prior"), ("bikes_available",), False),
        ModelSpec("persistence", "reference", Persistence, ("bikes_available",), False),
    ]


def ml_models(k: int, depth: int, knn_weights=None) -> list[ModelSpec]:
    """`knn_weights`: per-feature weights for the weighted k-NN (see pipeline.knn_weights), or None to skip it."""
    specs = [
        ModelSpec("naive Bayes", "ml", naive_bayes, tuple(FEATURES_NO_DOW), True),
        ModelSpec(f"k-NN (k={k})", "ml", partial(knn, k), tuple(FEATURES_NO_DOW), True),
        ModelSpec(f"decision tree (depth={depth})", "ml", partial(tree, depth), tuple(FEATURES), True),
        ModelSpec("random forest", "ml", forest, tuple(FEATURES), True),
        ModelSpec("gradient boosting", "ml", boosting, tuple(FEATURES), True),
    ]
    if knn_weights is not None:
        specs.insert(2, ModelSpec(f"weighted k-NN (k={k})", "ml", partial(knn, k, weights=tuple(knn_weights)),
                                  tuple(FEATURES_NO_DOW), True))
    return specs
