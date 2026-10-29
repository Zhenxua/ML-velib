"""Steps shared by scripts/run_pipeline.py and the notebook.

Protocol (Question A):
  1. hyper-parameters (k of k-NN, depth of the tree) are tuned on validation; the weighted k-NN's
     feature weights are learned inside the training block;
  2. every model is fitted on train; ML models get an F1-optimal threshold chosen on validation;
  3. the final model is the ML model with the best validation PR-AUC; choices are frozen in selection.json;
  4. only then is the test week scored — once, with the frozen models and thresholds.
"""
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score

from .config import DATA_FILE, EMPTY_THRESHOLD, HORIZONS, SEED, STEP_MIN
from .data import load_raw
from .evaluation import best_f1_threshold, classification_metrics
from .features import FEATURE_GROUPS, FEATURES, FEATURES_NO_DOW, add_features
from .grid import to_regular_grid
from .models import ModelSpec, forest, knn, ml_models, references, tree
from .split import build_dataset, split_sets


@dataclass
class Fitted:
    spec: ModelSpec
    model: object
    threshold: float | None  # None: fixed decision rule (references)

    @property
    def name(self) -> str:
        return self.spec.name

    def inputs(self, d: pd.DataFrame) -> pd.DataFrame:
        return d[list(self.spec.features)]

    def score(self, d: pd.DataFrame):
        return self.model.predict_proba(self.inputs(d))[:, 1]

    def predict(self, d: pd.DataFrame, score=None):
        if self.threshold is None:
            return self.model.predict(self.inputs(d))
        score = self.score(d) if score is None else score
        return (score >= self.threshold).astype(int)


def prepare(path: Path = DATA_FILE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Raw snapshots and the regular grid with all features."""
    raw = load_raw(path)
    return raw, add_features(to_regular_grid(raw))


def build_all(grid: pd.DataFrame, closed_as_positive: bool = False) -> dict[int, dict[str, pd.DataFrame]]:
    """{horizon_min: {"train": ..., "val": ..., "test": ...}}"""
    return {h: split_sets(build_dataset(grid, steps, closed_as_positive)) for h, steps in HORIZONS.items()}


def tune_knn(sets: dict, ks=(5, 15, 50, 150), n_sub: int | None = None) -> tuple[int, dict]:
    """k maximising validation PR-AUC, with each candidate fitted on the full training block by default.
    The best k grows with the number of training rows, so tuning on a subsample and refitting on all rows
    is biased towards too small a k (on this data: k=15 on 50 000 rows, but k=50 is better on 219 000).
    `n_sub` (random rows of the training block only — no period mixing) is a speed option for exploration."""
    tr, va = sets["train"], sets["val"]
    sub = tr if n_sub is None else tr.sample(min(n_sub, len(tr)), random_state=SEED)
    scores = {}
    for k in ks:
        p = knn(k).fit(sub[FEATURES_NO_DOW], sub["y"]).predict_proba(va[FEATURES_NO_DOW])[:, 1]
        scores[k] = average_precision_score(va["y"], p)
    return max(scores, key=scores.get), scores


def tune_tree(sets: dict, depths=(4, 8, 12, 16)) -> tuple[int, dict]:
    tr, va = sets["train"], sets["val"]
    scores = {}
    for depth in depths:
        p = tree(depth).fit(tr[FEATURES], tr["y"]).predict_proba(va[FEATURES])[:, 1]
        scores[depth] = average_precision_score(va["y"], p)
    return max(scores, key=scores.get), scores


def inner_split(train: pd.DataFrame, steps: int, holdout_days: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the training block for anything that must be estimated "from train only" but needs held-out data:
    all but the last `holdout_days` days (fit part), then the last days starting one horizon after the cut
    (evaluation part), so that no label of the fit part — observed at t + h — falls inside the evaluation part."""
    t = train.index.get_level_values("t")
    cut = t.max() - pd.Timedelta(days=holdout_days)
    return train[t < cut], train[t >= cut + pd.Timedelta(minutes=STEP_MIN * steps)]


def knn_weights(sets: dict, steps: int, holdout_days: int = 7, n: int = 15_000) -> pd.Series:
    """Feature weights for the weighted k-NN, estimated inside the training block only (no validation data).

    A random forest is fitted on the fit part of `inner_split`; permutation importance (PR-AUC drop) is
    measured on its evaluation part. The station encoding st_empty_rate is recomputed from the fit part
    only, so the held-out days do not leak into it during this inner evaluation.
    Weights = sqrt(normalised positive importance), because the Euclidean distance squares each
    (standardised) coordinate.
    """
    fit_part, eval_part = inner_split(sets["train"], steps, holdout_days)
    rate = (fit_part["bikes_available"] < EMPTY_THRESHOLD).groupby(level="station_code").mean()

    def inputs(d: pd.DataFrame) -> pd.DataFrame:
        station_rate = rate.reindex(d.index.get_level_values("station_code")).to_numpy()
        return d[FEATURES_NO_DOW].assign(st_empty_rate=station_rate)

    model = forest().fit(inputs(fit_part), fit_part["y"])
    sample = eval_part.sample(min(n, len(eval_part)), random_state=SEED)
    imp = permutation_importance(model, inputs(sample), sample["y"], scoring="average_precision",
                                 n_repeats=3, random_state=SEED, n_jobs=-1)
    w = np.clip(imp.importances_mean, 0, None)
    if w.sum() == 0:  # no feature helps on the held-out days: fall back to the plain (unweighted) distance
        w = np.ones_like(w)
    return pd.Series(np.sqrt(w / w.sum()), index=FEATURES_NO_DOW)


def fit_and_validate(sets: dict, specs: list[ModelSpec]) -> tuple[list[Fitted], pd.DataFrame]:
    """Fit on train, choose the threshold on validation, report validation metrics.
    These validation numbers are *selection* numbers (optimistic by construction); the unbiased
    estimate is the test evaluation done after `choose`."""
    tr, va = sets["train"], sets["val"]
    fitted, rows = [], []
    for spec in specs:
        start = time.perf_counter()
        model = spec.make().fit(tr[list(spec.features)], tr["y"])
        f = Fitted(spec, model, None)
        score = f.score(va)
        if spec.tune_threshold:
            f.threshold = best_f1_threshold(va["y"], score)
        fitted.append(f)
        rows.append({"model": spec.name, "family": spec.family, "threshold": f.threshold,
                     **classification_metrics(va["y"], score, f.predict(va, score)),
                     "seconds": round(time.perf_counter() - start, 1)})
    return fitted, pd.DataFrame(rows)


def choose(val_table: pd.DataFrame) -> str:
    """Pre-registered rule: the ML model with the highest validation PR-AUC."""
    ml = val_table[val_table["family"] == "ml"]
    return ml.loc[ml["pr_auc"].idxmax(), "model"]


def select_models(data: dict) -> tuple[dict, pd.DataFrame, dict]:
    """Steps 1-3 of the protocol for every horizon, on train and validation only: tune k and the depth, learn
    the k-NN weights inside train, fit every model, set thresholds, choose. Returns the fitted models per
    horizon, the validation table and the selection record to freeze — the single schema of selection.json
    (one code path for the script and the notebook)."""
    fitted, tables, selection = {}, [], {}
    for h, sets in data.items():
        k, k_scores = tune_knn(sets)
        depth, depth_scores = tune_tree(sets)
        weights = knn_weights(sets, HORIZONS[h])
        fitted[h], val = fit_and_validate(sets, references() + ml_models(k, depth, weights))
        selection[h] = {"chosen_model": choose(val), "knn_k": k, "tree_depth": depth,
                        "knn_val_pr_auc": k_scores, "tree_val_pr_auc": depth_scores,
                        "knn_weights": weights.round(4).to_dict(),
                        "thresholds": {f.name: f.threshold for f in fitted[h] if f.threshold is not None}}
        tables.append(val.assign(horizon_min=h))
    return fitted, pd.concat(tables, ignore_index=True), selection


def save_selection(selection: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(selection, indent=2, default=float), encoding="utf-8")


def evaluate(fitted: list[Fitted], d: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Metrics of frozen models on a block, plus {name: (score, pred)} for the error analysis."""
    rows, outputs = [], {}
    for f in fitted:
        score = f.score(d)
        pred = f.predict(d, score)
        outputs[f.name] = (score, pred)
        rows.append({"model": f.name, "family": f.spec.family, "threshold": f.threshold,
                     **classification_metrics(d["y"], score, pred)})
    return pd.DataFrame(rows), outputs


def ablation(sets: dict) -> pd.DataFrame:
    """Real contribution of each feature group (brief: "apport réel des variables"): validation PR-AUC of a
    random forest refitted without that group, and with the current state only. The forest is used because it
    is stable across seeds (validation PR-AUC std 0.0004 over 5 seeds, against 0.008 for gradient boosting at
    30 min, which would drown deltas of 0.01). Unlike permutation importance, refitting lets correlated
    features stand in for each other: a small drop means the information is redundant, not useless."""
    tr, va = sets["train"], sets["val"]
    configs = {"all features": FEATURES}
    configs |= {f"without {name}": [f for f in FEATURES if f not in cols] for name, cols in FEATURE_GROUPS.items()}
    configs["current state only"] = FEATURE_GROUPS["state"]
    rows = []
    for label, feats in configs.items():
        p = forest().fit(tr[feats], tr["y"]).predict_proba(va[feats])[:, 1]
        rows.append({"configuration": label, "n_features": len(feats), "val_pr_auc": average_precision_score(va["y"], p)})
    out = pd.DataFrame(rows).set_index("configuration")
    out["delta_vs_all"] = out["val_pr_auc"] - out.loc["all features", "val_pr_auc"]
    return out
