"""Alignment of the snapshots on a regular (station x 15-min slot) grid.

Why a regular grid: on it, `groupby(station).shift(k)` means *exactly* k x 15 min earlier
for the same station, even across collection outages (the missing slots exist as NaN rows).
Without it, shift(k) would silently jump over outages and over stations missing from partial snapshots.
"""
import numpy as np
import pandas as pd

from .config import FREQ, MAX_ALIGN_OFFSET

STATIC_COLUMNS = ["name", "commune", "insee_code", "latitude", "longitude", "capacity"]


def to_regular_grid(raw: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (station_code, t) on a regular 15-min UTC grid.

    - Snapshots more than MAX_ALIGN_OFFSET away from a quarter hour are dropped (in this file: only the
      first one, 2026-08-25 11:57:26 UTC). All the others are collected 1-7 s after a quarter hour,
      so rounding moves them by seconds, never across a slot (asserted).
    - Slots without an observation (outages, stations absent from a partial snapshot) stay NaN:
      no interpolation and no forward fill, so no value is ever borrowed from another time.
    - Static station attributes are filled on every slot. `capacity` is constant per station
      (asserted), so filling its 26 missing snapshots is not a look-ahead.
    - `closed` is the *observed* closure at that slot only (is_renting == 0), with no look-ahead.
    """
    df = raw.copy()
    slot = df["collected_at_utc"].dt.round(FREQ)
    aligned = (df["collected_at_utc"] - slot).abs() <= MAX_ALIGN_OFFSET
    df = df[aligned].assign(t=slot[aligned])
    if df.duplicated(["station_code", "t"]).any():
        raise ValueError("two snapshots of the same station fall in the same 15-min slot")

    if df.groupby("station_code")["capacity"].nunique().max() != 1:
        raise ValueError("capacity varies within a station: it cannot be treated as a static attribute")
    static = df.groupby("station_code")[STATIC_COLUMNS].first()  # first non-null value per station

    stations = np.sort(df["station_code"].unique())
    slots = pd.date_range(df["t"].min(), df["t"].max(), freq=FREQ)
    index = pd.MultiIndex.from_product([stations, slots], names=["station_code", "t"])
    g = df.drop(columns=STATIC_COLUMNS).set_index(["station_code", "t"]).reindex(index)

    station_of_row = g.index.get_level_values("station_code")
    for col in STATIC_COLUMNS:
        g[col] = static[col].reindex(station_of_row).to_numpy()
    g["observed"] = g["bikes_available"].notna()
    g["closed"] = g["is_renting"].eq(0)  # NaN (unobserved) -> False; such rows are excluded via `observed`
    return g


def n_unaligned(raw: pd.DataFrame) -> int:
    """Number of raw rows dropped by `to_regular_grid` because they are not aligned with a quarter hour."""
    slot = raw["collected_at_utc"].dt.round(FREQ)
    return int(((raw["collected_at_utc"] - slot).abs() > MAX_ALIGN_OFFSET).sum())
