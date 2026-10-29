"""Chronological train / validation / test split (no shuffling anywhere)."""
import numpy as np
import pandas as pd

from .config import SPLITS, STEP_MIN, T_TEST, T_VAL
from .target import make_target

EXCLUSION_REASONS = ["not observed at t", "closed at t", "not observed at t+h", "closed at t+h",
                     "t+h in the next block", "kept"]


def assign_split(t: pd.DatetimeIndex) -> np.ndarray:
    return np.select([t < T_VAL, t < T_TEST], ["train", "val"], default="test")


def build_dataset(g: pd.DataFrame, steps: int, closed_as_positive: bool = False) -> pd.DataFrame:
    """Samples (features at t, label at t + h) with their block.

    A sample is kept only if t and t + h fall in the same block: otherwise the last training rows
    would carry labels observed during validation (and validation labels from the test week).
    """
    y, valid = make_target(g, steps, closed_as_positive)
    d = g[valid].assign(y=y)
    t = d.index.get_level_values("t")
    d["set"] = assign_split(t)
    same_block = d["set"].to_numpy() == assign_split(t + pd.Timedelta(minutes=STEP_MIN * steps))
    return d[same_block]


def split_sets(d: pd.DataFrame) -> dict[str, pd.DataFrame]:
    sets = {name: d[d["set"] == name] for name in SPLITS}
    t = {name: s.index.get_level_values("t") for name, s in sets.items()}
    if not (t["train"].max() < t["val"].min() and t["val"].max() < t["test"].min()):
        raise ValueError("train / validation / test blocks overlap in time")
    return sets


def population_table(g: pd.DataFrame, steps: int) -> pd.DataFrame:
    """Every grid slot attributed to the *first* reason that excludes it (in the order of EXCLUSION_REASONS),
    by block of t. The 'kept' row equals the sizes of the train / val / test sets used for modelling."""
    by_station = g.groupby(level="station_code", sort=False)
    future_observed = by_station["observed"].shift(-steps, fill_value=False).to_numpy()
    future_closed = by_station["closed"].shift(-steps, fill_value=False).to_numpy()
    t = g.index.get_level_values("t")
    block = assign_split(t)
    crosses = block != assign_split(t + pd.Timedelta(minutes=STEP_MIN * steps))
    reason = np.select([~g["observed"].to_numpy(), g["closed"].to_numpy(), ~future_observed, future_closed, crosses],
                       EXCLUSION_REASONS[:-1], default="kept")
    table = pd.crosstab(pd.Categorical(reason, categories=EXCLUSION_REASONS),
                        pd.Categorical(block, categories=list(SPLITS)), dropna=False)
    table.index.name, table.columns.name = "reason", "block of t"
    table["total"] = table.sum(axis=1)
    return table


def summary(sets: dict[str, pd.DataFrame], show_test_labels: bool = False) -> pd.DataFrame:
    """Size, period and positive rate of each block. Before the model is chosen, the test block's positive
    rate is hidden (show_test_labels=False): nothing about the test labels is looked at until then."""
    rows = {}
    for name, s in sets.items():
        hidden = name == "test" and not show_test_labels
        rows[name] = {"samples": len(s), "positive rate": "(hidden)" if hidden else round(s["y"].mean(), 4),
                      "from (UTC)": s.index.get_level_values("t").min(), "to (UTC)": s.index.get_level_values("t").max()}
    return pd.DataFrame(rows).T
