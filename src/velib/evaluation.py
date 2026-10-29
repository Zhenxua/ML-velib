"""Metrics for a rare target (~11 % positives): PR-AUC, precision, recall, F1 and the confusion
matrix — accuracy is deliberately not reported (always predicting "available" already scores 87-90 %)."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve

from .config import EMPTY_THRESHOLD, LOCAL_TZ, SEED

HOUR_BANDS = {"night 0-7h": (0, 7), "morning peak 7-10h": (7, 10), "day 10-17h": (10, 17),
              "evening peak 17-20h": (17, 20), "evening 20-24h": (20, 24)}
SITUATIONS = ["stays available", "becomes empty", "stays empty", "recovers"]


def best_f1_threshold(y, score) -> float:
    """Threshold maximising F1. Call it on the validation set only; the value is then frozen for test."""
    precision, recall, thresholds = precision_recall_curve(y, score)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(thresholds[np.argmax(f1[:-1])])


def precision_at_recall(y, score, recall_target: float) -> float:
    """Best precision a score reaches while keeping recall >= recall_target. Used to compare a model with
    persistence *at persistence's own recall*, i.e. at a matched operating point rather than over the whole
    PR curve (where the coarse persistence score is at a disadvantage)."""
    precision, recall, _ = precision_recall_curve(y, score)
    return float(precision[recall >= recall_target].max())


def classification_metrics(y, score, pred) -> dict:
    """Precision is NaN (undefined) when nothing is predicted positive, e.g. for the majority class."""
    y, pred = np.asarray(y).astype(int), np.asarray(pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    both_classes = 0 < y.sum() < len(y)
    return {"n": len(y), "positives": int(y.sum()), "prevalence": y.mean(),
            "pr_auc": average_precision_score(y, score) if both_classes else np.nan,
            "precision": precision, "recall": recall, "f1": f1,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


def scored_frame(d: pd.DataFrame, score, pred) -> pd.DataFrame:
    """y / score / prediction of one model, with the grouping variables used in the error analysis."""
    t_local = d.index.get_level_values("t").tz_convert(LOCAL_TZ)
    hour = np.asarray(t_local.hour)
    band = np.select([(hour >= lo) & (hour < hi) for lo, hi in HOUR_BANDS.values()], list(HOUR_BANDS), default="")
    now_empty = d["bikes_available"].to_numpy() < EMPTY_THRESHOLD
    y = d["y"].to_numpy()
    situation = np.select(
        [~now_empty & (y == 0), ~now_empty & (y == 1), now_empty & (y == 1), now_empty & (y == 0)],
        SITUATIONS, default="")
    return pd.DataFrame({
        "y": y, "score": np.asarray(score), "pred": np.asarray(pred).astype(int),
        "station_code": d.index.get_level_values("station_code"),
        "area": np.where(d["paris"].to_numpy() == 1, "Paris", "suburbs"),
        "hour_local": hour,
        "hour_band": pd.Categorical(band, categories=list(HOUR_BANDS), ordered=True),
        "day_type": np.where(np.asarray(t_local.dayofweek) >= 5, "weekend", "weekday"),
        "day": pd.Categorical(t_local.date),
        "situation": pd.Categorical(situation, categories=SITUATIONS),
    }, index=d.index)


def metrics_by(scored: pd.DataFrame, by: str) -> pd.DataFrame:
    # group by the column itself: "station_code" is also an index level of scored frames
    return pd.DataFrame({key: classification_metrics(g["y"], g["score"], g["pred"])
                         for key, g in scored.groupby(scored[by], observed=True)}).T


def daily_dispersion(scored: pd.DataFrame) -> pd.DataFrame:
    """Day-by-day metrics (Paris calendar days) and their mean / std / min / max: a dispersion measure
    that respects time (each day is a block)."""
    daily = metrics_by(scored, "day")[["n", "prevalence", "pr_auc", "precision", "recall", "f1"]].astype(float)
    return pd.concat([daily, daily.agg(["mean", "std", "min", "max"])])


def station_comparison(scored_model: pd.DataFrame, scored_reference: pd.DataFrame) -> pd.DataFrame:
    """Per-station F1 and PR-AUC of a model and of a reference on the same rows, sorted from the station where
    the model loses most to the one where it gains most (PR-AUC is NaN for stations without any positive)."""
    cols = ["n", "positives", "pr_auc", "f1"]
    m = metrics_by(scored_model, "station_code")[cols].astype(float)
    r = metrics_by(scored_reference, "station_code")[["pr_auc", "f1"]].astype(float)
    out = m.join(r, rsuffix="_reference")
    out["f1_gain"] = out["f1"] - out["f1_reference"]
    return out.sort_values("f1_gain")


def situation_table(scored_by_model: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Share of samples predicted "empty" in each situation. Persistence is right by construction when the
    state does not change and wrong on every change, so a model's added value lies in the transitions.
    For 'becomes empty' / 'stays empty' the share is a recall; for the other two it is a false-alarm rate."""
    first = next(iter(scored_by_model.values()))
    out = first.groupby("situation", observed=True).size().rename("n").to_frame()
    out["share of samples"] = out["n"] / out["n"].sum()
    for name, s in scored_by_model.items():
        out[f"predicted empty: {name}"] = s.groupby("situation", observed=True)["pred"].mean()
    return out


def station_bootstrap(scored_a: pd.DataFrame, scored_b: pd.DataFrame | None = None, labels=("model", "persistence"),
                      n_boot: int = 500, seed: int = SEED) -> pd.DataFrame:
    """95 % percentile intervals of PR-AUC and F1 obtained by resampling *stations* with replacement.

    Rows of one station are strongly autocorrelated, so resampling rows would understate the
    uncertainty; stations are the independent-ish units. With `scored_b` (same rows, another model),
    the paired difference a - b is computed on the same resampled stations.
    """
    if scored_b is not None and not (scored_a.index.equals(scored_b.index)
                                     and np.array_equal(scored_a["y"].to_numpy(), scored_b["y"].to_numpy())):
        raise ValueError("paired bootstrap needs both models scored on the same rows, in the same order")
    rng = np.random.default_rng(seed)
    stations = scored_a["station_code"].to_numpy()
    codes = np.unique(stations)
    rows_of = {c: np.flatnonzero(stations == c) for c in codes}
    frames = {labels[0]: scored_a} if scored_b is None else {labels[0]: scored_a, labels[1]: scored_b}
    draws = []
    for _ in range(n_boot):
        idx = np.concatenate([rows_of[c] for c in rng.choice(codes, size=len(codes), replace=True)])
        draw = {}
        for label, s in frames.items():
            m = classification_metrics(s["y"].to_numpy()[idx], s["score"].to_numpy()[idx], s["pred"].to_numpy()[idx])
            draw[f"pr_auc ({label})"], draw[f"f1 ({label})"] = m["pr_auc"], m["f1"]
        if scored_b is not None:
            a, b = labels
            draw[f"pr_auc ({a} - {b})"] = draw[f"pr_auc ({a})"] - draw[f"pr_auc ({b})"]
            draw[f"f1 ({a} - {b})"] = draw[f"f1 ({a})"] - draw[f"f1 ({b})"]
        draws.append(draw)
    draws = pd.DataFrame(draws)
    return pd.DataFrame({"mean": draws.mean(), "ci_low": draws.quantile(0.025), "ci_high": draws.quantile(0.975)})
