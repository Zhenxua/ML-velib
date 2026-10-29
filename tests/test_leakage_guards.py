"""Guards against the pitfalls listed in the brief ("Pièges à éviter") and against leakage in the protocol.

    python tests/test_leakage_guards.py      # no extra dependency
    python -m pytest tests                   # if pytest is installed

Synthetic data is used so that each test isolates one rule; the last test checks the real file
(skipped if data/ has not been downloaded yet).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from velib.config import DATA_FILE, LOCAL_TZ, STEP_MIN, T_TEST, T_VAL  # noqa: E402
from velib.evaluation import best_f1_threshold, station_bootstrap  # noqa: E402
from velib.features import add_features  # noqa: E402
from velib.grid import to_regular_grid  # noqa: E402
from velib.models import Persistence, ml_models, references  # noqa: E402
from velib.pipeline import choose, fit_and_validate, inner_split  # noqa: E402
from velib.split import build_dataset, population_table, split_sets  # noqa: E402
from velib.target import make_target  # noqa: E402

UTC = "UTC"


def synthetic_raw(start="2026-09-20 00:00", end="2026-10-03 00:00", drop=(), closed=(), bikes_override=None):
    """Two stations: station 1 cycles through 0..4 bikes (often 'empty'), station 2 through 10..14.
    `drop` / `closed`: iterables of (station, UTC timestamp string) to remove / mark is_renting=0.
    `bikes_override`: optional function (station, t) -> bikes or None."""
    times = pd.date_range(start, end, freq=f"{STEP_MIN}min", tz=UTC)
    drop = {(s, pd.Timestamp(t, tz=UTC)) for s, t in drop}
    closed = {(s, pd.Timestamp(t, tz=UTC)) for s, t in closed}
    rows = []
    for station, base, commune in ((1, 0, "Paris"), (2, 10, "Montreuil")):
        for i, t in enumerate(times):
            if (station, t) in drop:
                continue
            bikes = base + i % 5
            if bikes_override is not None and bikes_override(station, t) is not None:
                bikes = bikes_override(station, t)
            is_open = 0 if (station, t) in closed else 1
            rows.append({
                "collected_at_utc": t + pd.Timedelta(seconds=2), "source_updated_at": t - pd.Timedelta(minutes=30),
                "station_code": station, "name": f"S{station}", "commune": commune, "insee_code": 75056,
                "latitude": 48.85, "longitude": 2.35, "capacity": 20.0, "bikes_available": bikes,
                "mechanical_bikes": bikes, "electric_bikes": 0, "docks_available": 20 - bikes,
                "is_installed": 1, "is_renting": is_open, "is_returning": is_open,
            })
    return pd.DataFrame(rows)


def ts(s):
    return pd.Timestamp(s, tz=UTC)


def station_rows(g, station):
    return g.index.get_level_values("station_code") == station


# --- pitfalls of the brief ------------------------------------------------------------------------------

def test_lags_never_cross_stations():
    g = add_features(to_regular_grid(synthetic_raw()))
    first = g.index.get_level_values("t").min()
    # First slot of station 2: there is no earlier row *of station 2*, so every lag must be NaN
    # (an ungrouped shift would read station 1's last row here).
    for col in ("d_bikes_15", "d_bikes_30", "d_bikes_60", "bikes_24h"):
        assert np.isnan(g.loc[(2, first), col]), col
    t = ts("2026-09-22 10:00")
    expected = g.loc[(2, t), "bikes_available"] - g.loc[(2, t - pd.Timedelta(minutes=15)), "bikes_available"]
    assert g.loc[(2, t), "d_bikes_15"] == expected


def test_collection_gaps_are_not_filled():
    gap = "2026-09-22 10:00"
    g = add_features(to_regular_grid(synthetic_raw(drop=[(1, gap)])))
    assert not g.loc[(1, ts(gap)), "observed"]
    assert np.isnan(g.loc[(1, ts(gap)), "bikes_available"])  # no interpolation / forward fill
    assert np.isnan(g.loc[(1, ts(gap) + pd.Timedelta(minutes=15)), "d_bikes_15"])  # lag of a missing slot stays NaN
    _, valid = make_target(g, steps=2)
    assert not valid.loc[(1, ts(gap) - pd.Timedelta(minutes=30))]  # label time unobserved -> no sample


def test_lags_skip_closed_slots():
    # A closed station mostly reports 0 bikes: a lag that points into a closure must be NaN, not 0.
    closed_slot = ts("2026-09-22 10:00")
    g = add_features(to_regular_grid(synthetic_raw(closed=[(1, "2026-09-22 10:00")])))
    after = closed_slot + pd.Timedelta(minutes=15)
    assert np.isnan(g.loc[(1, after), "d_bikes_15"])
    assert np.isnan(g.loc[(1, closed_slot + pd.Timedelta(days=1)), "bikes_24h"])
    assert not np.isnan(g.loc[(1, after), "d_bikes_30"])  # the other lags still read open slots


def test_calendar_features_use_paris_time():
    g = add_features(to_regular_grid(synthetic_raw(start="2026-09-25 00:00", end="2026-10-27 00:00")))
    for utc, local_hour in (("2026-09-26 06:00", 8), ("2026-10-26 06:00", 7)):  # UTC+2, then UTC+1 after DST
        row = g.loc[(1, ts(utc))]
        assert np.isclose(row["hour_sin"], np.sin(2 * np.pi * local_hour / 24))
        assert np.isclose(row["hour_cos"], np.cos(2 * np.pi * local_hour / 24))
    late = ts("2026-09-26 23:00")  # Saturday 23:00 UTC = Sunday 01:00 in Paris
    assert g.loc[(1, late), "dow"] == late.tz_convert(LOCAL_TZ).dayofweek == 6
    assert g.loc[(1, late), "weekend"] == 1


def test_target_is_same_station_h_ahead():
    g = to_regular_grid(synthetic_raw())
    for steps in (2, 4):
        y, valid = make_target(g, steps)
        for station in (1, 2):
            t = ts("2026-09-23 12:00")
            future = g.loc[(station, t + pd.Timedelta(minutes=STEP_MIN * steps)), "bikes_available"]
            assert y.loc[(station, t)] == int(future < 2)
    assert (y.xs(2, level="station_code") == 0).all()  # station 2 never has < 2 bikes


def test_closure_flag_has_no_lookahead():
    # Closed at t-15 and t+15 but open at t: the original notebook's `closed | (prev & next)` marked t as
    # closed using the *next* slot. The causal flag keeps t as a valid prediction time.
    t = "2026-09-22 10:00"
    closed = [(1, "2026-09-22 09:45"), (1, "2026-09-22 10:15")]
    g = to_regular_grid(synthetic_raw(closed=closed))
    assert not g.loc[(1, ts(t)), "closed"]
    _, valid = make_target(g, steps=4)
    assert valid.loc[(1, ts(t))]
    # Sensitivity variant: a closure at t + h counts as "departure impossible".
    y_pos, valid_pos = make_target(g, steps=1, closed_as_positive=True)
    assert valid_pos.loc[(1, ts(t))] and y_pos.loc[(1, ts(t))] == 1


def test_split_is_chronological_and_labels_stay_in_block():
    g = add_features(to_regular_grid(synthetic_raw()))
    for steps in (2, 4):
        sets = split_sets(build_dataset(g, steps))
        horizon = pd.Timedelta(minutes=STEP_MIN * steps)
        t = {name: s.index.get_level_values("t") for name, s in sets.items()}
        assert (t["train"] + horizon < T_VAL).all()
        assert ((t["val"] >= T_VAL) & (t["val"] + horizon < T_TEST)).all()
        assert (t["test"] >= T_TEST).all()


# --- statistics learned from data ----------------------------------------------------------------------

def test_station_statistic_is_fitted_on_train_only():
    base = add_features(to_regular_grid(synthetic_raw()))
    # Make station 2 empty during validation/test only: no station rate may move.
    emptied = add_features(to_regular_grid(synthetic_raw(bikes_override=lambda s, t: 0 if s == 2 and t >= T_VAL else None)))
    assert np.allclose(base["st_empty_rate"], emptied["st_empty_rate"], equal_nan=True)


def test_station_rate_is_one_constant_per_station():
    # A leave-one-week-out encoding would let the trees decode each row's own-week labels through station
    # identity (latitude, longitude): the rate must be one train-period constant per station, on every row.
    g = add_features(to_regular_grid(synthetic_raw(start="2026-09-01 00:00")))
    assert (g.groupby(level="station_code")["st_empty_rate"].nunique() == 1).all()
    assert g.loc[station_rows(g, 1), "st_empty_rate"].iloc[0] > 0  # station 1 is often empty
    assert g.loc[station_rows(g, 2), "st_empty_rate"].iloc[0] == 0  # station 2 never is


def test_inner_split_of_train_is_purged():
    g = add_features(to_regular_grid(synthetic_raw(start="2026-09-01 00:00")))
    for steps in (2, 4):
        sets = split_sets(build_dataset(g, steps))
        fit_part, eval_part = inner_split(sets["train"], steps)
        last_label = fit_part.index.get_level_values("t").max() + pd.Timedelta(minutes=STEP_MIN * steps)
        assert last_label < eval_part.index.get_level_values("t").min()
        assert eval_part.index.get_level_values("t").max() < T_VAL


# --- selection protocol --------------------------------------------------------------------------------

def test_choices_ignore_test_labels():
    g = add_features(to_regular_grid(synthetic_raw()))
    sets = split_sets(build_dataset(g, 2))
    specs = references() + [s for s in ml_models(5, 3) if s.name.startswith("decision tree")]
    _, val = fit_and_validate(sets, specs)
    flipped = {**sets, "test": sets["test"].assign(y=1 - sets["test"]["y"])}
    _, val_flipped = fit_and_validate(flipped, specs)
    pd.testing.assert_frame_equal(val.drop(columns="seconds"), val_flipped.drop(columns="seconds"))
    assert choose(val) == choose(val_flipped)


def test_best_f1_threshold_is_optimal():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    score = np.round(rng.random(500) + 0.3 * y, 2)  # many ties

    def f1(cut):
        pred = score >= cut
        return 2 * (pred & (y == 1)).sum() / (pred.sum() + y.sum())

    assert np.isclose(f1(best_f1_threshold(y, score)), max(f1(c) for c in np.unique(score)))


def test_population_table_matches_the_modelling_sets():
    g = add_features(to_regular_grid(synthetic_raw(drop=[(1, "2026-09-22 10:00")], closed=[(2, "2026-09-26 08:00")])))
    for steps in (2, 4):
        table = population_table(g, steps)
        sets = split_sets(build_dataset(g, steps))
        assert table.loc["kept", ["train", "val", "test"]].tolist() == [len(sets[b]) for b in ("train", "val", "test")]
        assert table["total"].sum() == len(g)


def test_paired_bootstrap_needs_aligned_frames():
    rows = pd.DataFrame({"y": [0, 1] * 20, "score": np.linspace(0, 1, 40), "pred": [0, 1] * 20,
                         "station_code": np.repeat(np.arange(4), 10)})
    try:
        station_bootstrap(rows, rows.sample(frac=1, random_state=0), n_boot=5)
    except ValueError:
        pass
    else:
        raise AssertionError("misaligned frames were accepted")
    same = station_bootstrap(rows, rows.copy(), n_boot=20)
    assert np.allclose(same.loc["pr_auc (model - persistence)"], 0)


def test_persistence_reference():
    X = pd.DataFrame({"bikes_available": [0, 1, 2, 7]})
    model = Persistence().fit(X)
    assert model.predict(X).tolist() == [1, 1, 0, 0]
    assert ((model.predict_proba(X)[:, 1] >= 0.5).astype(int) == model.predict(X)).all()


# --- real data -----------------------------------------------------------------------------------------

def test_real_data_regression():
    if not DATA_FILE.exists():
        try:
            import pytest
        except ImportError:
            print("  (skipped: data file not downloaded)")
            return
        pytest.skip("data file not downloaded")
    from velib.pipeline import build_all, prepare
    raw, grid = prepare()
    assert raw["station_code"].nunique() == 80 and len(raw) == 329_680
    assert grid.groupby(level="station_code").size().nunique() == 1  # same regular grid for every station
    data = build_all(grid)
    sizes = {"train": 219_127, "val": 47_612, "test": 59_574}
    assert {k: len(v) for k, v in data[30].items()} == sizes
    assert population_table(grid, 2).loc["kept", list(sizes)].tolist() == list(sizes.values())


if __name__ == "__main__":
    tests = [f for name, f in globals().items() if name.startswith("test_") and callable(f)]
    for test in tests:
        test()
        print("ok ", test.__name__)
    print(f"{len(tests)} tests passed")
