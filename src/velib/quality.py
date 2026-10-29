"""Data-quality description of the raw snapshots.

The brief asks that outages, closures and incomplete snapshots be *described, not hidden*.
Every function here only measures; none of them modifies the data.
"""
import pandas as pd


def coverage(df: pd.DataFrame) -> pd.Series:
    per_station = df.groupby("station_code").size()
    return pd.Series({
        "rows": len(df),
        "stations": df["station_code"].nunique(),
        "snapshots": df["collected_at_utc"].nunique(),
        "first_snapshot_utc": df["collected_at_utc"].min(),
        "last_snapshot_utc": df["collected_at_utc"].max(),
        "rows_per_station_min": per_station.min(),
        "rows_per_station_max": per_station.max(),
        "duplicated_station_snapshot": int(df.duplicated(["station_code", "collected_at_utc"]).sum()),
    })


def missing_values(df: pd.DataFrame) -> pd.Series:
    na = df.isna().sum()
    return na[na > 0]


def consistency(df: pd.DataFrame) -> pd.Series:
    """Counts of rows that contradict the declared capacity (NaN capacity is counted separately)."""
    total = df["bikes_available"] + df["docks_available"]
    return pd.Series({
        "bikes != mechanical + electric": int((df["bikes_available"] != df["mechanical_bikes"] + df["electric_bikes"]).sum()),
        "bikes + docks > capacity": int((total > df["capacity"]).sum()),
        "bikes + docks < capacity (unavailable docks)": int((total < df["capacity"]).sum()),
        "capacity missing": int(df["capacity"].isna().sum()),
        "bikes + docks == 0": int((total == 0).sum()),
    })


def over_capacity_by_station(df: pd.DataFrame, top: int = 5) -> pd.Series:
    over = df[df["bikes_available"] + df["docks_available"] > df["capacity"]]
    return over.groupby("station_code").size().sort_values(ascending=False).head(top)


def collection_gaps(df: pd.DataFrame, min_gap: str = "20min") -> pd.DataFrame:
    """Intervals between consecutive snapshots that exceed `min_gap` (nominal step: 15 min)."""
    snaps = pd.Series(sorted(df["collected_at_utc"].unique()))
    minutes = snaps.diff().dt.total_seconds() / 60
    gaps = pd.DataFrame({"start": snaps.shift(1), "end": snaps, "minutes": minutes})
    return gaps[minutes > pd.Timedelta(min_gap).total_seconds() / 60].sort_values("minutes", ascending=False)


def partial_snapshots(df: pd.DataFrame) -> pd.DataFrame:
    """Snapshots that do not contain every station, with their share of Paris stations."""
    n_stations = df["station_code"].nunique()
    per_snap = df.groupby("collected_at_utc").agg(
        n_rows=("station_code", "size"),
        share_paris=("commune", lambda s: s.eq("Paris").mean()),
    )
    return per_snap[per_snap["n_rows"] < n_stations]


def partial_snapshot_sets(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Distinct station sets among the incomplete snapshots, and whether those sets are disjoint and together
    cover every station. On this file there is one 30-station set and one 50-station set, complementary: the
    incomplete snapshots are always the same block of stations missing, not a random subset (and not a Paris /
    suburbs split: both sets mix the two areas)."""
    partial = partial_snapshots(df).index
    sets = df[df["collected_at_utc"].isin(partial)].groupby("collected_at_utc")["station_code"].apply(frozenset)
    distinct = sorted(set(sets), key=len)
    paris = set(df.loc[df["commune"].eq("Paris"), "station_code"])
    table = pd.DataFrame([{"stations in the set": len(s), "snapshots": int(sets.map(lambda x: x == s).sum()),
                           "of which Paris": len(s & paris)} for s in distinct])
    union = frozenset().union(*distinct)
    complementary = sum(map(len, distinct)) == len(union) == df["station_code"].nunique()
    return table, complementary


def closure_episodes(df: pd.DataFrame) -> pd.DataFrame:
    """Runs of is_renting == 0 per station.

    An episode starts at a closed row whose previous *observed* row (same station) was open,
    so a station missing from a partial snapshot does not split an episode in two.
    """
    closed = df["is_renting"].eq(0)
    prev_closed = closed.groupby(df["station_code"]).shift(1, fill_value=False)
    episode = (closed & ~prev_closed).groupby(df["station_code"]).cumsum()
    return (df[closed].assign(episode=episode[closed])
            .groupby(["station_code", "episode"])["collected_at_utc"]
            .agg(start="min", end="max", n_snapshots="size")
            .reset_index())


def source_staleness_minutes(df: pd.DataFrame) -> pd.Series:
    """collected_at_utc - source_updated_at, in minutes (descriptive only; not used as a feature)."""
    lag = (df["collected_at_utc"] - df["source_updated_at"]).dt.total_seconds() / 60
    return lag.describe(percentiles=[0.5, 0.9, 0.99])


def unchanged_source_but_bikes_changed(df: pd.DataFrame) -> int:
    """Rows whose source timestamp did not move while the bike count did: source_updated_at is not a reliable freshness signal."""
    g = df.groupby("station_code")
    same_src = g["source_updated_at"].diff() == pd.Timedelta(0)
    bikes_moved = g["bikes_available"].diff().fillna(0) != 0
    return int((same_src & bikes_moved).sum())


def static_stations(df: pd.DataFrame) -> pd.DataFrame:
    """Per station: share of consecutive collected snapshots in which bikes *and* docks are unchanged, and the
    longest unchanged run (in snapshots). A station that almost never changes may be frozen in the feed
    rather than quiet — the two cannot be told apart from the counts alone (cf. Question E of the brief)."""
    g = df.groupby("station_code")
    same = (g["bikes_available"].diff() == 0) & (g["docks_available"].diff() == 0)
    run = (~same).groupby(df["station_code"]).cumsum()
    longest = same.groupby([df["station_code"], run]).sum().groupby(level="station_code").max() + 1
    out = pd.DataFrame({"unchanged_share": same.groupby(df["station_code"]).mean(), "longest_unchanged_run": longest})
    return out.sort_values("unchanged_share", ascending=False)


def report(df: pd.DataFrame) -> dict:
    """All quality tables in one dict (used by the script and the notebook)."""
    return {
        "static_stations": static_stations(df),
        "coverage": coverage(df),
        "missing_values": missing_values(df),
        "consistency": consistency(df),
        "over_capacity_by_station": over_capacity_by_station(df),
        "collection_gaps": collection_gaps(df),
        "partial_snapshots": partial_snapshots(df),
        "partial_snapshot_sets": partial_snapshot_sets(df),
        "closure_episodes": closure_episodes(df),
        "source_staleness_minutes": source_staleness_minutes(df),
        "unchanged_source_but_bikes_changed": unchanged_source_but_bikes_changed(df),
    }
