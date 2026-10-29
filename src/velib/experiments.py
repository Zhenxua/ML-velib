"""Exploratory analyses (the first notebook's cells 31-32, fixed, plus a seed check). They use the
training and validation blocks only — never the test week."""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import SEED
from .features import FEATURES, FEATURES_NO_DOW
from .models import boosting, forest


def correlated_pairs(train: pd.DataFrame, features=FEATURES_NO_DOW, threshold: float = 0.7) -> pd.Series:
    """Feature pairs with |Pearson r| > threshold on the training block (redundancy check)."""
    corr = train[features].corr()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    pairs = upper.stack()
    return pairs[pairs.abs() > threshold].sort_values(key=abs, ascending=False)


def permutation_importance_val(fitted, val: pd.DataFrame, n: int = 15_000, repeats: int = 5) -> pd.DataFrame:
    """PR-AUC drop when one feature is shuffled, on a validation sample, for a model fitted on train.
    Correlated features (bikes_available / fill / bikes_24h) share the credit, so a small value does
    not mean "useless" — compare with pipeline.ablation, which refits without each group."""
    sample = val.sample(min(n, len(val)), random_state=SEED)
    X = fitted.inputs(sample)
    imp = permutation_importance(fitted.model, X, sample["y"], scoring="average_precision",
                                 n_repeats=repeats, random_state=SEED, n_jobs=-1)
    return (pd.DataFrame({"mean": imp.importances_mean, "std": imp.importances_std}, index=X.columns)
            .sort_values("mean", ascending=False))


def seed_sensitivity(sets: dict, seeds=range(5)) -> pd.DataFrame:
    """Validation PR-AUC of the two ensembles over several random seeds. Random forests draw bootstrap
    samples and HistGradientBoosting bins features from a random 200 000-row subsample when the training
    block is larger (219 000 rows here): when two models are close, one seed is not enough to rank them."""
    tr, va = sets["train"], sets["val"]
    rows = []
    for seed in seeds:
        for name, make in (("random forest", forest), ("gradient boosting", boosting)):
            p = make(seed).fit(tr[FEATURES], tr["y"]).predict_proba(va[FEATURES])[:, 1]
            rows.append({"model": name, "seed": seed, "val_pr_auc": average_precision_score(va["y"], p)})
    return pd.DataFrame(rows).groupby("model")["val_pr_auc"].agg(["mean", "std", "min", "max"])


def pca_val(sets: dict, n_components=(3, 6, 10), k: int = 50) -> pd.DataFrame:
    """Naive Bayes and k-NN on the first principal components (scaler and PCA fitted on train)."""
    tr, va = sets["train"], sets["val"]
    rows = []
    for n in n_components:
        for name, clf in (("naive Bayes", GaussianNB()), (f"k-NN (k={k})", KNeighborsClassifier(n_neighbors=k, n_jobs=-1))):
            pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                 PCA(n_components=n, random_state=SEED), clf)
            p = pipe.fit(tr[FEATURES_NO_DOW], tr["y"]).predict_proba(va[FEATURES_NO_DOW])[:, 1]
            rows.append({"model": name, "components": n, "val_pr_auc": average_precision_score(va["y"], p)})
    return pd.DataFrame(rows)
