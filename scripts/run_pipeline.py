"""Reproduce every number of the project (Question A) from the frozen dataset.

    python scripts/run_pipeline.py            # main protocol, test evaluation, sensitivity, ablation
    python scripts/run_pipeline.py --extras   # + feature correlations, permutation importance,
                                              #   5-seed sensitivity, PCA (validation only)

Writes reports/results/*.csv|json and reports/figures/*.png; the elapsed time is printed for every section.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless: figures are only saved to reports/figures
import pandas as pd  # noqa: E402

from velib import evaluation as ev  # noqa: E402
from velib import experiments, plots, quality  # noqa: E402
from velib.config import FIGURES_DIR, HORIZONS, RESULTS_DIR, ROOT  # noqa: E402
from velib.grid import n_unaligned  # noqa: E402
from velib.pipeline import ablation, build_all, evaluate, prepare, save_selection, select_models  # noqa: E402
from velib.split import population_table, summary  # noqa: E402

COLS = ["model", "pr_auc", "precision", "recall", "f1", "tp", "fp", "fn", "tn", "threshold"]
GROUP_COLS = ["n", "prevalence", "pr_auc", "precision", "recall", "f1"]
SUSPECT_STATION = 45504  # almost never changes state (see quality.static_stations)
pd.set_option("display.width", 180)
pd.set_option("display.max_columns", 30)
START = time.perf_counter()


def section(title: str) -> None:
    elapsed = f"[{(time.perf_counter() - START) / 60:.1f} min]"
    print(f"\n{'=' * 8} {title} {'=' * max(4, 70 - len(title))} {elapsed}", flush=True)


def fmt(scores: dict) -> str:
    return ", ".join(f"{k}: {v:.3f}" for k, v in scores.items())


def main(extras: bool, n_boot: int) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    section("1. Data and quality (described, not hidden)")
    raw, grid = prepare()
    q = quality.report(raw)
    for key in ("coverage", "missing_values", "consistency", "over_capacity_by_station", "source_staleness_minutes"):
        print(f"\n-- {key}\n{q[key]}")
    print(f"\n-- collection gaps > 20 min: {len(q['collection_gaps'])}\n{q['collection_gaps'].head(5).to_string(index=False)}")
    sets_table, complementary = q["partial_snapshot_sets"]
    print(f"\n-- incomplete snapshots: {len(q['partial_snapshots'])}; distinct station sets:\n"
          f"{sets_table.to_string(index=False)}\n"
          f"   disjoint and covering all {raw['station_code'].nunique()} stations: {complementary}")
    print(f"\n-- closure episodes (is_renting == 0)\n{q['closure_episodes'].to_string(index=False)}")
    static = q["static_stations"]
    print(f"\n-- most static stations: share of consecutive snapshots with bikes and docks unchanged "
          f"(median over the 80 stations: {static['unchanged_share'].median():.2f})\n{static.head(5).round(3)}")
    print(f"\n-- rows whose source timestamp did not change while the bike count did: {q['unchanged_source_but_bikes_changed']}")
    print(f"-- raw rows dropped because not aligned on a quarter hour: {n_unaligned(raw)}")
    q["closure_episodes"].to_csv(RESULTS_DIR / "closure_episodes.csv", index=False)
    q["collection_gaps"].to_csv(RESULTS_DIR / "collection_gaps.csv", index=False)

    section("2. Population and chronological split (test labels not looked at)")
    data = build_all(grid)
    for h, steps in HORIZONS.items():
        population = population_table(grid, steps)
        print(f"\nhorizon {h} min — each grid slot counted under the first reason that excludes it\n{population}")
        population.to_csv(RESULTS_DIR / f"population_{h}min.csv")
        print(summary(data[h]))
        nan_share = pd.DataFrame({b: s["bikes_24h"].isna().mean() for b, s in data[h].items()}, index=["bikes_24h missing"])
        print(nan_share.round(3))

    section("3. Model selection on validation (the test week is not touched)")
    fitted, val, selection = select_models(data)
    for h in HORIZONS:
        s = selection[h]
        print(f"\nhorizon {h} min | k-NN val PR-AUC by k: {fmt(s['knn_val_pr_auc'])} -> k={s['knn_k']}"
              f" | tree by depth: {fmt(s['tree_val_pr_auc'])} -> depth={s['tree_depth']}"
              f"\nweighted k-NN, largest weights (learned on train only): "
              f"{fmt(dict(sorted(s['knn_weights'].items(), key=lambda kv: -kv[1])[:6]))}")
        print(val[val["horizon_min"] == h][COLS + ["seconds"]].round(3).to_string(index=False))
        print(f"-> chosen on validation PR-AUC: {s['chosen_model']}", flush=True)
    val.to_csv(RESULTS_DIR / "val_metrics.csv", index=False)
    save_selection(selection, RESULTS_DIR / "selection.json")  # frozen *before* the test week is scored

    section("4. Final evaluation on the test week (frozen models and thresholds)")
    test_tables, panels, matched = [], {}, []
    for h, sets in data.items():
        te, chosen = sets["test"], selection[h]["chosen_model"]
        print(f"\nhorizon {h} min — test block\n{summary({'test': te}, show_test_labels=True)}")
        table, outputs = evaluate(fitted[h], te)
        table.insert(0, "horizon_min", h)
        test_tables.append(table)
        print(table[COLS].round(3).to_string(index=False))

        pers = table.set_index("model").loc["persistence"]
        matched.append({"horizon_min": h, "model": chosen, "persistence recall": pers["recall"],
                        "persistence precision": pers["precision"],
                        "model precision at that recall": ev.precision_at_recall(te["y"], outputs[chosen][0], pers["recall"])})
        print(f"-- matched operating point: {pd.DataFrame([matched[-1]]).round(3).to_string(index=False)}")

        scored = {name: ev.scored_frame(te, *outputs[name]) for name in (chosen, "persistence")}
        for by in ("area", "hour_band", "day_type"):
            groups = pd.concat({name: ev.metrics_by(s, by)[GROUP_COLS] for name, s in scored.items()}, names=["model", by])
            print(f"\n-- by {by}\n{groups.astype(float).round(3)}")
            groups.to_csv(RESULTS_DIR / f"test_by_{by}_{h}min.csv")
        daily = pd.concat({name: ev.daily_dispersion(s) for name, s in scored.items()}, names=["model", "day"])
        print(f"\n-- day-by-day dispersion\n{daily.round(3)}")
        daily.to_csv(RESULTS_DIR / f"test_daily_{h}min.csv")
        boot = ev.station_bootstrap(scored[chosen], scored["persistence"], labels=(chosen, "persistence"), n_boot=n_boot)
        print(f"\n-- station bootstrap ({n_boot} resamples of the 80 stations), 95 % intervals\n{boot.round(3)}")
        boot.to_csv(RESULTS_DIR / f"test_station_bootstrap_{h}min.csv")
        stations = ev.station_comparison(scored[chosen], scored["persistence"])
        with_pos = stations[stations["positives"] > 0]
        print(f"\n-- per station: {chosen} F1 >= persistence F1 on {(with_pos['f1_gain'] >= 0).mean():.0%} of the "
              f"{len(with_pos)} stations with at least one positive; largest losses:\n{with_pos.head(5).round(3)}")
        stations.to_csv(RESULTS_DIR / f"test_by_station_{h}min.csv")
        situations = ev.situation_table(scored)
        print(f"\n-- situations (share predicted 'empty')\n{situations.round(3)}")
        situations.to_csv(RESULTS_DIR / f"test_situations_{h}min.csv")

        panels[h] = {"y": te["y"].to_numpy(), "model": (chosen, *outputs[chosen]), "persistence": outputs["persistence"]}
        plots.by_hour(chosen, scored[chosen], scored["persistence"], h, FIGURES_DIR / f"by_hour_test_{h}min.png")
        plots.situations(situations, chosen, h, FIGURES_DIR / f"situations_test_{h}min.png")
    pd.concat(test_tables).to_csv(RESULTS_DIR / "test_metrics.csv", index=False)
    pd.DataFrame(matched).to_csv(RESULTS_DIR / "test_matched_operating_point.csv", index=False)
    plots.pr_curves(panels, FIGURES_DIR / "pr_curves_test.png")

    section("5. Sensitivity of the test results (frozen models)")
    data_closed = build_all(grid, closed_as_positive=True)
    rows = []
    for h in HORIZONS:
        keep = [f for f in fitted[h] if f.name in {selection[h]["chosen_model"], "persistence"}]
        te = data[h]["test"]
        variants = {"closure at t+h counted as 'departure impossible'": data_closed[h]["test"],
                    f"without station {SUSPECT_STATION}": te[te.index.get_level_values("station_code") != SUSPECT_STATION]}
        for variant, d in variants.items():
            table, _ = evaluate(keep, d)
            rows.append(table.assign(horizon_min=h, variant=variant, n_samples=len(d), n_default=len(te)))
    sensitivity = pd.concat(rows)
    print(sensitivity[["horizon_min", "variant", "n_samples", "n_default"] + COLS].round(3).to_string(index=False))
    sensitivity.to_csv(RESULTS_DIR / "test_sensitivity.csv", index=False)

    section("6. Contribution of feature groups (validation; random forest refitted without each group)")
    for h, sets in data.items():
        abl = ablation(sets)
        print(f"\nhorizon {h} min\n{abl.round(4)}", flush=True)
        abl.to_csv(RESULTS_DIR / f"val_ablation_{h}min.csv")

    if extras:
        section("7. Extra experiments (validation only)")
        print(f"-- |r| > 0.7 on train\n{experiments.correlated_pairs(data[30]['train']).round(2)}")
        for h in HORIZONS:
            chosen_model = next(f for f in fitted[h] if f.name == selection[h]["chosen_model"])
            imp = experiments.permutation_importance_val(chosen_model, data[h]["val"])
            print(f"\n-- permutation importance ({chosen_model.name}, {h} min, validation sample)\n{imp.round(4)}")
            seeds = experiments.seed_sensitivity(data[h])
            print(f"\n-- seed sensitivity, {h} min (validation PR-AUC over 5 seeds)\n{seeds.round(4)}", flush=True)
            seeds.to_csv(RESULTS_DIR / f"val_seed_sensitivity_{h}min.csv")
        pca = experiments.pca_val(data[30], k=selection[30]["knn_k"])
        print(f"\n-- PCA + naive Bayes / k-NN, 30 min\n{pca.round(3)}")
        pca.to_csv(RESULTS_DIR / "val_pca_30min.csv", index=False)

    section("Done")
    print(f"Tables in {RESULTS_DIR.relative_to(ROOT).as_posix()}/, figures in {FIGURES_DIR.relative_to(ROOT).as_posix()}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--extras", action="store_true", help="also run the exploratory experiments (slower)")
    parser.add_argument("--n-boot", type=int, default=500, help="station bootstrap resamples (default 500)")
    args = parser.parse_args()
    main(args.extras, args.n_boot)
