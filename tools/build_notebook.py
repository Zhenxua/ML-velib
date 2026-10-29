"""Generate notebooks/velib_question_A.ipynb (nbformat 4.5) from the cell list below, then execute it with
tools/execute_notebook.py to store the outputs:

    python tools/build_notebook.py                     # -> notebooks/velib_question_A.ipynb
    python tools/execute_notebook.py notebooks/velib_question_A.ipynb

Section 11 (conclusions) is read from tools/conclusions.md. If you edit the .ipynb by hand afterwards, the
notebook becomes the source of truth: do not regenerate it.
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CELLS = []


def md(text):
    CELLS.append(("markdown", text.strip("\n")))


def code(text):
    CELLS.append(("code", text.strip("\n")))


md(r"""
# Vélib' sous tension — Question A: *departure impossible*

**Question.** For each of the 80 stations, predict **30 and 60 minutes ahead** whether the station will have
**fewer than 2 bikes available** (`bikes_available < 2`, "departure impossible"). One common model for all stations.
References: the majority class and **persistence** (the future state equals the current state).

**Protocol.** Strictly chronological split — train (25 Aug → the 28 h collection outage of 23–24 Sep),
validation (24 Sep 13:45 UTC → 1 Oct 00:00 Paris time), test (1 → 8 Oct). Hyper-parameters, decision
thresholds and the final model are chosen on validation only and frozen before the test week is scored, once.
Every preprocessing step is fitted on train; every lag is computed per station on a regular 15-min grid;
calendar features use Paris local time.

All the logic lives in `src/velib/` (one module per step) and is covered by `tests/test_leakage_guards.py`; this
notebook is the commented walk-through and calls the same functions as `scripts/run_pipeline.py`, which writes the
reference tables to `reports/results/`. The notebook writes its own figures to `notebooks/figures/` only.
""")

code(r"""
import sys
from pathlib import Path

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "velib").is_dir())
sys.path.insert(0, str(ROOT / "src"))

import json

import numpy as np
import pandas as pd
from IPython.display import Image, display

from velib import evaluation as ev
from velib import experiments, plots, quality
from velib.config import HORIZONS, LOCAL_TZ
from velib.data import fetch, md5sum
from velib.features import FEATURES
from velib.grid import n_unaligned
from velib.pipeline import ablation, build_all, evaluate, prepare, select_models
from velib.split import population_table, summary

pd.set_option("display.precision", 3)
pd.set_option("display.width", 160)
FIGURES = ROOT / "notebooks" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
RUN_EXTRAS = False  # True: also run the appendix experiments (several more minutes)
GROUP_COLS = ["n", "prevalence", "pr_auc", "precision", "recall", "f1"]
SUSPECT_STATION = 45504
""")

md(r"""
## 1. Data: download once, verify the md5

The frozen file is not versioned. `fetch()` downloads it only if `data/` does not contain it yet, and always checks
its md5 against the published one (`02ab1e3fdac10579e4ce93d1540a5dc7`) before anything else.
""")

code(r"""
path = fetch()
print(path.name, md5sum(path))
raw, grid = prepare(path)  # raw snapshots, and the regular (station x 15 min) grid with all features
q = quality.report(raw)
q["coverage"]
""")

md(r"""
## 2. Data quality: described, not hidden

The brief asks that outages, closures and incoherent values be counted and described. Nothing below modifies the data.
""")

code(r"""
display(q["missing_values"].to_frame("missing"), q["consistency"].to_frame("rows"),
        q["over_capacity_by_station"].to_frame("rows with bikes + docks > capacity"))
""")

md(r"""
* `capacity` is missing in 26 snapshots, but it is **constant within each station** (checked in `grid.py`),
  so it is a static attribute and filling it is not a look-ahead.
* `bikes + docks > capacity` is concentrated on a few stations (45504 alone accounts for ~73 %): the declared
  capacity is unreliable there, which is why the fill ratio uses `bikes / (bikes + docks)` rather than `capacity`.
* `bikes + docks < capacity` in about half of the rows simply reflects docks out of service.
""")

code(r"""
q["collection_gaps"]
""")

code(r"""
sets_table, complementary = q["partial_snapshot_sets"]
print(f"{len(q['partial_snapshots'])} incomplete snapshots; distinct station sets disjoint and covering the 80 stations: {complementary}")
sets_table
""")

md(r"""
* One **28-hour outage** (23 Sep 09:45 → 24 Sep 13:45 UTC) and nine 30-minute gaps (one missed snapshot each).
* 36 **incomplete snapshots**: they contain either always the same 30 stations or always the complementary 50.
  The same block of stations is missing each time, so this is a truncated collection, not a sample. It is *not* a
  Paris / suburbs split either (both sets mix the two areas), so the hypothesis tested in the first version of the
  notebook is rejected.
* On the regular grid, a station missing from a snapshot is simply an **unobserved slot (NaN)**: no interpolation,
  no forward fill, and lags that point to it are NaN.
""")

code(r"""
q["closure_episodes"]
""")

code(r"""
static = q["static_stations"]
print(f"median share of unchanged consecutive snapshots over the 80 stations: {static['unchanged_share'].median():.2f}")
static.head(5)
""")

code(r"""
print("collected_at_utc - source_updated_at (minutes):")
display(q["source_staleness_minutes"].to_frame().T)
print("rows whose source timestamp did not change while the bike count did:", q["unchanged_source_but_bikes_changed"])
""")

md(r"""
* Four stations were **closed** (`is_renting = is_returning = 0`) for a total of 812 rows, in long episodes (1–2 days).
  A closure is a different phenomenon from running out of bikes: samples from a closed station are excluded from
  the population and **counted** (section 3), and section 8 checks that the choice does not matter. A closed station
  mostly reports 0 bikes, so closed slots are not used as lag sources either (section 5).
* Station 45504 almost never changes state and has most of the `bikes + docks > capacity` rows: it may be frozen in
  the feed. It never has fewer than 2 bikes, so it only adds easy negatives. It is **kept and described**, and
  section 8 shows the test results without it.
* `source_updated_at` lags the collection by ~1 h and often does not move while the bike count does: it is not a
  reliable freshness signal and is not used as a feature.
""")

md(r"""
## 3. Regular grid, exact target and population

* Grid: one row per (station, 15-min UTC slot) — 80 × 4,258 slots. Only the first snapshot (11:57 UTC, not aligned)
  is dropped; all others are collected 1–7 s after a quarter hour.
* **Target**: `y(station, t) = 1` if `bikes_available(station, t + h) < 2`, with `h` = 30 or 60 min, read with a
  per-station `shift(-h)` on the grid (same station, exactly `h` later).
* **Population**: the station is observed and open at `t`, observed and open at `t + h`, and `t + h` falls in the same
  block as `t`. Each grid slot is counted below under the first reason that excludes it.
""")

code(r"""
print("raw rows dropped because not aligned on a quarter hour:", n_unaligned(raw))
for h, steps in HORIZONS.items():
    print(f"horizon {h} min")
    display(population_table(grid, steps))
""")

md(r"""
Target distribution. The breakdowns by hour, area and station use the **training block only**. The weekly series
uses train and validation, never the test week, so nothing about the test labels informs the modelling choices.
""")

code(r"""
data = build_all(grid)
tr30 = data[30]["train"]
hour = tr30.index.get_level_values("t").tz_convert(LOCAL_TZ).hour  # Paris local time
display(tr30.groupby(hour)["y"].mean().rename("positive rate by hour (Paris), train").to_frame().T)
display(tr30.groupby(np.where(tr30["paris"] == 1, "Paris", "suburbs"))["y"].mean().rename("positive rate, train").to_frame())
display(tr30.groupby(level="station_code")["y"].mean().describe().rename("positive rate per station, train").to_frame().T)
trval = pd.concat([data[30]["train"], data[30]["val"]])
week = trval.index.get_level_values("t").tz_convert(LOCAL_TZ).isocalendar().week.to_numpy()
trval.groupby(week)["y"].mean().rename("positive rate by ISO week (train + validation)").to_frame().T
""")

md(r"""
## 4. Chronological split

A sample is kept only if `t` and `t + h` fall in the same block, so no training label is observed during validation
and no validation label during the test week. The test block's positive rate stays hidden until the model is chosen.
The training block includes the end of the summer holidays and the positive rate drifts upward week after week
(section 3): a non-stationarity to keep in mind.

`bikes_24h` is missing much more often on validation, because its first day follows the 28 h outage.
""")

code(r"""
for h, sets in data.items():
    print(f"horizon {h} min")
    display(summary(sets))
pd.DataFrame({b: s["bikes_24h"].isna().mean() for b, s in data[30].items()}, index=["bikes_24h missing (30 min)"])
""")

md(r"""
## 5. Features (all available at prediction time t)

| group | features | how |
|---|---|---|
| state | `bikes_available`, `docks_available`, `fill` | current snapshot; `fill = bikes / (bikes + docks)` |
| trend | `d_bikes_15/30/60`, `bikes_24h` | per-station `shift(k)` on the regular grid, from open slots only (NaN across gaps and closures) |
| calendar | `hour_sin/cos`, `dow`, `weekend` | **Paris local time** (`tz_convert("Europe/Paris")`) |
| station | `capacity`, `paris`, `latitude`, `longitude`, `st_empty_rate` | static; `st_empty_rate` = share of empty slots on the training block, one constant per station |

`st_empty_rate` is deliberately not computed "out of fold". A leave-one-week-out rate is, within a station, almost the
mirror image of that week's own empty rate. Since the trees can also identify the station from its coordinates, they
would decode each training row's own-week labels, a pattern that does not exist on new weeks.

k-NN and naive Bayes do not use `dow`: coded 0–6 it would be read as an ordered distance (Sunday "far" from Monday).
""")

code(r"""
display(tr30[FEATURES].isna().mean().rename("missing share (train)").to_frame().T)
tr30.groupby(pd.cut(tr30["bikes_available"], [-1, 0, 1, 2, 4, 8, 100]), observed=True)["y"].mean().rename(
    "P(y = 1 | bikes now), train").to_frame().T
""")

md(r"""
## 6. Model selection on validation (the test week is not touched)

`select_models` runs the same code as the script. For each horizon:
* `k` (k-NN) and the depth (tree) maximise the validation PR-AUC;
* the weighted k-NN uses feature weights learned **inside the training block** (permutation importance of a forest,
  with the last training week held out);
* each ML model gets the F1-optimal threshold on validation;
* the final model is the ML model with the best validation PR-AUC.

These validation figures are **selection** figures, optimistic by construction. The choices are then frozen.
""")

code(r"""
fitted, val, selection = select_models(data)
for h in HORIZONS:
    print(f"horizon {h} min — chosen on validation: {selection[h]['chosen_model']}")
    display(val[val["horizon_min"] == h].drop(columns=["family", "n", "positives", "prevalence", "horizon_min"]).set_index("model"))
print(json.dumps({h: {k: v for k, v in s.items() if k != "knn_weights"} for h, s in selection.items()}, indent=1, default=float))
""")

md(r"""
## 7. Final evaluation on the test week (frozen models and thresholds)
""")

code(r"""
outputs = {}
for h, sets in data.items():
    te = sets["test"]
    display(summary({"test": te}, show_test_labels=True))
    table, outputs[h] = evaluate(fitted[h], te)
    print(f"horizon {h} min — test week")
    display(table.drop(columns=["family", "n", "positives", "prevalence"]).set_index("model"))
    pers = table.set_index("model").loc["persistence"]
    chosen = selection[h]["chosen_model"]
    at_recall = ev.precision_at_recall(te["y"], outputs[h][chosen][0], pers["recall"])
    print(f"matched operating point: at persistence's recall ({pers['recall']:.3f}), precision {pers['precision']:.3f} "
          f"for persistence vs {at_recall:.3f} for {chosen}")
""")

code(r"""
panels = {h: {"y": data[h]["test"]["y"].to_numpy(),
              "model": (selection[h]["chosen_model"], *outputs[h][selection[h]["chosen_model"]]),
              "persistence": outputs[h]["persistence"]} for h in HORIZONS}
Image(plots.pr_curves(panels, FIGURES / "pr_curves_test.png"), width=900)
""")

md(r"""
Persistence's curve ranks samples by the current number of bikes (fewer bikes = higher risk). This graded score is
generous to the reference: with the binary rule alone, its PR-AUC would be much lower.

### Results by group, dispersion and uncertainty

* **By area, hour band (Paris time) and weekday / weekend.**
* **Day by day** on the test week: a dispersion measure that respects time.
* **Station bootstrap**: the 80 stations are resampled with replacement (rows of one station are dependent, so
  resampling rows would understate uncertainty). The paired difference model − persistence is computed on the same
  resampled stations.
* **Per station**: where the model loses to persistence.
""")

code(r"""
scored = {}
for h in HORIZONS:
    te, chosen = data[h]["test"], selection[h]["chosen_model"]
    scored[h] = {name: ev.scored_frame(te, *outputs[h][name]) for name in (chosen, "persistence")}
    print(f"===== horizon {h} min =====")
    for by in ("area", "hour_band", "day_type"):
        display(pd.concat({name: ev.metrics_by(s, by)[GROUP_COLS] for name, s in scored[h].items()},
                          names=["model", by]).astype(float))
""")

code(r"""
for h in HORIZONS:
    chosen = selection[h]["chosen_model"]
    print(f"===== horizon {h} min =====")
    display(pd.concat({name: ev.daily_dispersion(s) for name, s in scored[h].items()}, names=["model", "day"]))
    display(ev.station_bootstrap(scored[h][chosen], scored[h]["persistence"], labels=(chosen, "persistence")))
    stations = ev.station_comparison(scored[h][chosen], scored[h]["persistence"])
    with_pos = stations[stations["positives"] > 0]
    print(f"{chosen} F1 >= persistence F1 on {(with_pos['f1_gain'] >= 0).mean():.0%} of the {len(with_pos)} stations "
          "with at least one positive; largest losses:")
    display(with_pos.head(5))
""")

md(r"""
### Characteristic errors

Persistence is right by construction whenever the state does not change, and wrong on every change. Splitting the test
samples by *situation* (state now → state at `t + h`) shows where a model adds value: anticipating stations that
**become empty**, at the price of false alarms on stations that stay available.
""")

code(r"""
for h in HORIZONS:
    chosen = selection[h]["chosen_model"]
    table = ev.situation_table(scored[h])
    display(table)
    display(Image(plots.situations(table, chosen, h, FIGURES / f"situations_test_{h}min.png"), width=760))
    display(Image(plots.by_hour(chosen, scored[h][chosen], scored[h]["persistence"], h,
                                FIGURES / f"by_hour_test_{h}min.png"), width=880))
""")

md(r"""
## 8. Sensitivity of the test results (frozen models)

* A closure at `t + h` counted as "departure impossible" instead of being excluded.
* Station 45504, which may be frozen in the feed, removed.
""")

code(r"""
data_closed = build_all(grid, closed_as_positive=True)
for h in HORIZONS:
    keep = [f for f in fitted[h] if f.name in {selection[h]["chosen_model"], "persistence"}]
    te = data[h]["test"]
    variants = {"closure at t+h counted as positive": data_closed[h]["test"],
                f"without station {SUSPECT_STATION}": te[te.index.get_level_values("station_code") != SUSPECT_STATION]}
    for variant, d in variants.items():
        table, _ = evaluate(keep, d)
        print(f"horizon {h} min — {variant}: {len(d)} samples (default {len(te)})")
        display(table.drop(columns=["family"]).set_index("model"))
""")

md(r"""
## 9. What the features really bring (ablation, validation)

A random forest is refitted without each feature group. The forest is used because it is stable across seeds;
gradient boosting's seed noise is as large as the smaller effects. Unlike permutation importance, refitting lets
correlated features stand in for each other: a small drop means *redundant*, not *useless*.
""")

code(r"""
for h, sets in data.items():
    print(f"horizon {h} min")
    display(ablation(sets))
""")

md(r"""
## 10. Appendix: exploratory experiments (validation only)

Redundant feature pairs, permutation importance of the chosen model (mean ± std over 5 repeats), sensitivity of the
two ensembles to the random seed (when two models are within a few thousandths of PR-AUC, one seed cannot rank
them), and PCA + naive Bayes / k-NN. Set `RUN_EXTRAS = True` in the first cell to run them
(`python scripts/run_pipeline.py --extras` does the same).
""")

code(r"""
if RUN_EXTRAS:
    display(experiments.correlated_pairs(tr30).to_frame("r (train)"))
    for h in HORIZONS:
        chosen_model = next(f for f in fitted[h] if f.name == selection[h]["chosen_model"])
        print(f"horizon {h} min — permutation importance of {chosen_model.name} (validation sample)")
        display(experiments.permutation_importance_val(chosen_model, data[h]["val"]))
        print(f"horizon {h} min — validation PR-AUC over 5 seeds")
        display(experiments.seed_sensitivity(data[h]))
    display(experiments.pca_val(data[30], k=selection[30]["knn_k"]))
else:
    print("skipped (RUN_EXTRAS = False)")
""")

md(r"""
## 11. Conclusions and limits

__CONCLUSIONS__
""")


def build(path: Path, conclusions: str) -> None:
    cells = []
    for i, (kind, text) in enumerate(CELLS):
        text = text.replace("__CONCLUSIONS__", conclusions.strip("\n"))
        cell = {"cell_type": kind, "id": f"cell-{i:02d}", "metadata": {}, "source": text.splitlines(keepends=True)}
        if kind == "code":
            cell.update({"execution_count": None, "outputs": []})
        cells.append(cell)
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", nargs="?", type=Path, default=ROOT / "notebooks" / "velib_question_A.ipynb")
    parser.add_argument("--conclusions", type=Path, default=HERE / "conclusions.md")
    args = parser.parse_args()
    build(args.output, args.conclusions.read_text(encoding="utf-8"))
    print(f"wrote {args.output}")
