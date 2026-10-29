# Audit and refactoring report — Vélib' sous tension, Question A

**Scope.** The original notebook `ML Project velib-2.ipynb` (33 cells, Deepnote export) checked against the brief
`velib-sous-tension.pdf`.

**Method.**
1. A line-by-line review.
2. Four independent audits (split protocol and leakage · time grid, lags and time zones · data cleaning and target ·
   evaluation and reproducibility). An adversarial verifier re-measured every claim on the real data.
3. A refactor into a tested package, followed by a second independent review of the refactor (leakage and
   correctness · code quality and reproducibility · coverage of the audit), again with adversarial verification.
4. A final full re-run.

## 1. Verdict

The core methodology is **sound**: none of the six pitfalls listed in the brief is present (section 2). The split is
chronological, every lag is a per-station shift on a regular grid, local time goes through `Europe/Paris`, nothing is
interpolated, and the metrics suit a rare target.

The weaknesses lie elsewhere:

1. **The test week is never scored.** Every number in the notebook is a validation number, measured on the same week
   that was used to choose `k`, the tree depth, the PCA size, each decision threshold and the "best" model.
2. **Several items of the brief's *socle attendu* are missing**: results by group or period, a dispersion measure,
   error analysis, a real measure of feature contribution, and limits.
3. **The notebook is not reproducible.** The last cell was saved with a `NameError`, there is hidden state across
   cells, and the repository material is missing (README, pinned environment, licence, data attribution).
4. A few **small look-aheads and bugs**, each with a negligible effect on the numbers.

All of them are fixed in the refactored code. The headline results on the test week are:

| horizon | chosen on validation | PR-AUC model / persistence | F1 model / persistence | Δ PR-AUC [95 % station bootstrap] | Δ F1 | precision at persistence's recall |
|---|---|---|---|---|---|---|
| 30 min | random forest | **0.816** / 0.702 | **0.742** / 0.729 | +0.114 [0.101; 0.128] | +0.013 [0.007; 0.020] | 0.749 vs 0.729 |
| 60 min | gradient boosting | **0.754** / 0.619 | **0.689** / 0.660 | +0.134 [0.118; 0.150] | +0.030 [0.019; 0.040] | 0.700 vs 0.660 |

## 2. The brief's pitfalls ("Pièges à éviter")

| Pitfall | Original notebook | Evidence | Refactor |
|---|---|---|---|
| `train_test_split` with `shuffle=True` | **Absent** | `which_set()` splits on timestamps (cell-24). `tr.sample(50_000)` only subsamples training rows, and HGB early stopping is off. | `split.py`; `test_split_is_chronological_and_labels_stay_in_block` |
| Lags without `groupby(station)` | **Absent** | All shifts are grouped on a complete 80 × 4,258 grid. 0 lagged values come from another station (an ungrouped shift would contaminate 79 rows). | `features.py`; `test_lags_never_cross_stations` |
| UTC used as local time | **Absent** | `tz_convert("Europe/Paris")` is used everywhere and `T_TEST` is midnight Paris time. Using UTC would give a wrong weekday for 8.3 % of rows and put the noon peak at 10 h. | `features.py`; `test_calendar_features_use_paris_time` (also after DST) |
| Gaps interpolated with the next value | **Absent** for bikes and docks. Two negligible look-aheads: the closure flag `closed \| (prev & nxt)` (3 rows), and capacity filled with the whole-period max, which is harmless because capacity is constant per station. | No `ffill`/`interpolate` anywhere; 11,040 unobserved slots stay NaN. | Causal closure flag; capacity checked static; `test_collection_gaps_are_not_filled`, `test_closure_flag_has_no_lookahead` |
| Accuracy presented alone | **Absent** | PR-AUC, precision, recall, F1 and the confusion matrix are reported; accuracy never is. | Same, plus per-group, per-day, per-station and bootstrap results |
| Conclusions about users' demand | **Absent** | The notebook has almost no prose. | Conclusions (notebook §11) and the wording guide in section 6 |

Also checked and correct: imputers, scalers and PCA live inside pipelines fitted on train. Samples whose label falls
in the next block are dropped (160 rows at 30 min, 320 at 60 min). `st_empty_rate` is computed on train only. The
persistence reference is not handicapped: tuning its cut on validation still selects `bikes < 2`.

## 3. Findings on the original notebook, and fixes

Severity: **H** = changes what can be claimed / required by the brief, **M** = real weakness, **L** = minor.

| # | Sev. | Finding (original cells) | Evidence (re-measured) | Fix in the refactor |
|---|---|---|---|---|
| H1 | H | **Test week never scored**; every reported number is a selection number (cells 24–32). | Forest at 30 min: validation PR-AUC 0.828, test 0.816. Persistence drops as well (0.707 → 0.702), so the gap is mostly a change of period, not tuning optimism. | `pipeline.select_models` + `choose`; choices frozen in `results/selection.json` **before** the test week is scored once |
| H2 | H | **Missing *socle* / *Analyse* items**: no results by group or period, no dispersion, no error analysis (`probs` is stored but never used), no limits. | — | `evaluation.metrics_by` (area, hour band, weekday/weekend), `daily_dispersion`, `station_bootstrap`, `station_comparison`, `situation_table`, `precision_at_recall`; figures; conclusions |
| H3 | H | **Not reproducible.** Cell-32 was saved with a `NameError` from a fresh kernel, so the weighted k-NN has no result. Each re-run of cell-21 drops one more snapshot. A global `rows` list is patched with `drop_duplicates`. The data is re-downloaded on every run. There is no README, environment, licence or attribution. | Re-running the original code with scikit-learn 1.9.1 moves HGB's validation PR-AUC from 0.822 to 0.827 | `src/velib` package, `scripts/run_pipeline.py`, pinned `requirements.txt`, `tests/`, `README.md`, `LICENSE`, cached md5-checked download |
| M1 | M | The "best model" comes from point estimates without dispersion. | Over 5 seeds, HGB's validation PR-AUC ranges from 0.812 to 0.829, wider than the gap to the forest (0.828) | `experiments.seed_sensitivity`; random forest and boosting reported as **tied** (the 5-seed mean selects the same models) |
| M2 | M | k-NN `k` was tuned on a 50,000-row random subsample, then the model was refitted on 219,000 rows. | The best `k` grows with the number of rows. Tuned on the full block, k=50 gives 0.701 against 0.679 for the k=15 the subsample preferred | `pipeline.tune_knn` uses the full training block |
| M3 | M | Feature contribution relies only on permutation importance on validation (3 repeats, no std), with correlated features. | `bikes`/`fill` r = 0.80 and `bikes`/`bikes_24h` r = 0.78. Permutation gives `bikes_available` 0.46, but refitting without `bikes_available` alone costs only −0.013 PR-AUC | `pipeline.ablation` (drop-group refits of the seed-stable random forest); importance reported with std |
| M4 | M | The target population and its exclusions are not reported. | 812 rows closed at t; 12 (30 min) / 20 (60 min) closed at t+h; 2,303 / 2,503 unobserved at t+h; 160 / 320 crossing the val→test boundary | `split.population_table` (every slot by reason × block); sensitivity with a closure at t+h counted as positive: +5 / +8 test samples, metrics unchanged at 3 decimals |
| L1 | L | The closure flag `closed \| (prev & nxt)` uses the next slot (cell-21). | 3 observed open rows excluded | Causal flag in `grid.py`, plus a test |
| L2 | L | Closed slots are used as lag sources, and a closed station reports 0 bikes in 725 of its 812 closed rows. | `bikes_24h` = 0 on ≈480 samples the day after a closure | Lags read open slots only (`features.py`), plus a test |
| L3 | L | `st_empty_rate` counts closed rows (cell-26). | Station 4101: 0.045 vs 0.010 on open rows | Computed on open, observed slots of the training block |
| L4 | L | The target analysis uses all periods, test included (cell-21), and the test prevalence is printed before any choice. | Its rows are 67 % train, 15 % val, 18 % test | The notebook's target analysis uses train (and train + validation for the weekly series); the test rate is hidden until the choices are frozen |
| L5 | L | Station 45504 looks frozen and is not discussed. | 89 % unchanged consecutive snapshots (median station: 38 %), longest run 180 snapshots (≈45 h), 73 % of the `bikes+docks > capacity` rows | `quality.static_stations`; described. The test results without it are identical (it has no positive) |
| L6 | L | The validation week starts right after the 28 h outage. | `bikes_24h` is missing on 17 % of validation rows (4 % train, 1 % test) | Shown per block; stated as a limit |
| L7 | L | The data are non-stationary. | Positive rate by week rises from 8.6 % (wk 35, end of the holidays) to 12.8 % (wk 40) | Weekly table in the notebook; stated as a limit |
| L8 | L | The cell-18 hypothesis (30-row snapshots = suburbs) is refuted by its own output but left without a conclusion. | There is one 30-station set (17 in Paris) and one complementary 50-station set (33 in Paris) | `quality.partial_snapshot_sets`; described as the same block of stations missing |
| L9 | L | The ~62 min staleness of `source_updated_at` is flagged but not concluded. | It goes backwards 39,556 times within a station, and the bike count changes without the timestamp in 91,373 rows | Documented; not used as a feature |
| L10 | L | Bugs and style. Cell-1 prints `stations: 4139`: `iloc[:, 0]` is `collected_at_utc`, so that is the snapshot count. Typos (`ange`, `devide`, `raws`, `meanning`). The k-NN "reason below" for dropping `dow` is never given. `FunctionTransformer(lambda X: X*w)` cannot be pickled and depends on the global `w`. Cell-14 embeds an image that is not shipped. Comments mix Chinese, French and English. | — | `quality.coverage`; `FEATURES_NO_DOW` documented; `FeatureWeighter` class; English throughout |

## 4. Results: original vs refactor

PR-AUC / F1. Validation numbers are selection numbers; the test columns are the final estimates.

| model | 30 min · original (val) | 30 min · refactor (val) | **30 min · test** | 60 min · original (val) | 60 min · refactor (val) | **60 min · test** |
|---|---|---|---|---|---|---|
| majority class | 0.126 / 0 | 0.126 / 0 | 0.122 / 0 | 0.126 / 0 | 0.126 / 0 | 0.122 / 0 |
| persistence | 0.707 / 0.734 | 0.707 / 0.734 | 0.702 / 0.729 | 0.629 / 0.665 | 0.629 / 0.665 | 0.619 / 0.660 |
| naive Bayes | 0.626 / 0.638 | 0.626 / 0.639 | 0.629 / 0.634 | 0.562 / 0.587 | 0.562 / 0.588 | 0.561 / 0.569 |
| k-NN (k=50) | 0.702 / 0.650 | 0.701 / 0.651 | 0.699 / 0.642 | 0.671 / 0.626 | 0.670 / 0.626 | 0.662 / 0.619 |
| weighted k-NN | *NameError* | 0.805 / 0.739 | 0.799 / 0.736 | *NameError* | 0.738 / 0.679 | 0.730 / 0.677 |
| decision tree (depth 8) | 0.811 / 0.743 | 0.811 / 0.743 | 0.801 / 0.738 | 0.742 / 0.681 | 0.743 / 0.681 | 0.722 / 0.672 |
| random forest | 0.827 / 0.749 | 0.828 / 0.749 | **0.816 / 0.742** | 0.767 / 0.692 | 0.766 / 0.691 | 0.747 / 0.685 |
| gradient boosting | 0.822 / 0.747 | 0.823 / 0.746 | 0.812 / 0.744 | 0.764 / 0.691 | 0.767 / 0.693 | **0.754 / 0.689** |

Bold = model chosen on validation. The validation numbers reproduce the original within ±0.003. The differences come
from the scikit-learn version, the causal closure flag (+1 to +3 samples) and the lag masking. All other tables are in
`reports/results/`, the figures in `reports/figures/`, and the full printed output in `reports/results/run_log.txt`.
Those tables cover: area, hour band, weekday/weekend, day, station, the bootstrap, the situations, the sensitivity
analyses, the ablation, the seed study and PCA.

## 5. What changed in the code

| Module | Responsibility | Findings addressed |
|---|---|---|
| `config.py` | URL, md5, horizons, split boundaries, seed — written once | H3 |
| `data.py` | cached, atomic download + md5 check (with a remediation hint) + UTC parsing | H3, L10 |
| `quality.py` | descriptive checks: coverage, gaps, incomplete snapshots, closures, static stations, staleness | L5, L8, L9, L10 |
| `grid.py` | regular station × 15-min grid; causal closure flag; static attributes | L1 |
| `features.py` | per-station lags from open slots; Paris-time calendar; train-only station rate | L2, L3 |
| `target.py` | exact target, with the closure-as-positive variant | M4 |
| `split.py` | chronological blocks; labels never cross a boundary; population table; test labels hidden before selection | M4, L4 |
| `models.py` | references and the five model families + weighted k-NN, all as train-fitted pipelines | L10, H3 |
| `pipeline.py` | `select_models` (one code path for script and notebook): tuning, train-only k-NN weights with a purged inner split, thresholds on validation, frozen choice; ablation | H1, M2, M3 |
| `evaluation.py` | metrics, per-group / per-day / per-station results, matched operating point, station bootstrap, situations | H2 |
| `experiments.py` | redundancy, permutation importance ± std, seed sensitivity, PCA | M1, M3 |
| `scripts/run_pipeline.py` | one command → every table and figure (10–15 min with `--extras`) | H3 |
| `tests/test_leakage_guards.py` | 16 tests: the brief's pitfalls, train-only statistics, the inner purge, choices independent of test labels, threshold optimality, population counts, bootstrap pairing, real-data regression | all |
| `notebooks/velib_question_A.ipynb` | narrative walk-through, executed top to bottom; writes only to `notebooks/figures/` | H3 |

## 6. Recommendations for the report (6 pages) and the talk (10 minutes)

**Report outline (≈ page budget).**
1. *Question* (½ p): the exact target, population and horizons, and why 30 minutes is the minimum (at 15 minutes
   persistence is almost unbeatable).
2. *Data* (1 p): coverage; a quality table (outage, incomplete snapshots, closures, station 45504, capacity
   inconsistencies). One figure: the positive rate by Paris hour, on the training block.
3. *Protocol* (1 p): a timeline figure of train / validation / test with the outage; the leakage guards and the tests
   that enforce them; the frozen choice (`selection.json`); the disclosure in section 7.
4. *Results* (1½ p): the test table with both references and the matched operating point; the PR curves
   (`pr_curves_test.png`); bootstrap intervals; one per-group table (hour band).
5. *Errors and features* (1 p): `situations_test_30min.png` (the model anticipates ~21 % of new emptyings, persistence
   0 %); the ablation table.
6. *Limits and conclusion* (1 p): collection limits and limits of use (see the notebook's §11).

**Talk (≈ 8 slides).** Question → data and quality (one picture) → protocol timeline → headline result (PR curves)
→ where the model helps (situations) → when (by hour) → what matters (ablation) → limits and a one-sentence conclusion.

**Messages to state explicitly.**
* Validation numbers are selection numbers; **the test numbers are the result**.
* The gain over persistence is large in PR-AUC (ranking) and modest at persistence's own operating point (precision
  0.749 vs 0.729 at 30 min, 0.700 vs 0.660 at 60 min). It comes from anticipating new emptyings.
* The random forest and gradient boosting are tied (seed variability exceeds their gap); say "tree ensembles".
* Persistence's PR-AUC is computed with a graded score (fewer bikes = higher risk). With the binary rule it would be
  0.56 instead of 0.70 on test, so the comparison is generous to the reference.
* A methodological point worth a sentence: the station encoding is a single train-period constant per station, **not**
  "out of fold". A leave-one-week-out rate looked cleaner but leaked each training row's own-week labels through
  station identity (see section 8).

**Wording guide (no causal claims).**

| Avoid | Prefer |
|---|---|
| "Users take more bikes at noon, so stations empty." | "The share of slots with fewer than 2 bikes peaks around noon (Paris time) in the training weeks." |
| "The hour of day causes emptying." | "Adding calendar features raises validation PR-AUC by 0.02 (30 min) to 0.04 (60 min)." |
| "Demand is higher in Paris." | "Paris stations are more often observed with fewer than 2 bikes (test week: 13.3 % vs 10.3 % of samples)." |
| "The model predicts demand." | "The model predicts the state published by the feed 30 / 60 minutes ahead." |
| "Accuracy is 93 %." | Never report accuracy alone: the useless majority class already scores 87–90 %. |

**Possible improvements** (each to be decided on validation only, and only if time allows):
* *geographical neighbourhood* (an enrichment suggested by the brief): the mean fill ratio of the k nearest stations at t;
* rolling per-station mean and volatility over the past hour (on the grid, open slots only);
* a decision threshold set by costs, if a missed emptying is costlier than a false alarm;
* past observed weather (only values available at prediction time);
* a calibration check (reliability curve) before any probabilistic use;
* a training-window sensitivity study, given the drift.

**Before submitting.**
* Put your names in `LICENSE`.
* Choose the report language (code and comments are in English).
* Delete the two stale files in `reports/results/` (`population.csv`, `test_sensitivity_closures.csv`).
* Push with `data/` excluded (`.gitignore` already does it).
* Run `python tests/test_leakage_guards.py` and `python scripts/run_pipeline.py --extras`, then commit
  `reports/results/` and `reports/figures/`.

## 7. Disclosure: exposure of the test week

The final code scores the test week once, after the choices are frozen by fixed rules on validation. The test week was
nevertheless *seen* before that, twice:
1. The audit froze the **original** notebook's models and thresholds and scored them on the test week, to measure
   how optimistic the validation figures were (H1).
2. During development, one smoke check scored a depth-8 tree on the test week, to exercise the evaluation code. The
   script has been deleted.

Every change made afterwards follows from the brief's rules (causal closure flag, open-slot lags, train-only
statistics, hiding test labels) or from training/validation evidence (k tuned on the full block, seed study,
random-forest ablation). The selection rule (best validation PR-AUC) and the hyper-parameter grids are unchanged
from the original notebook. The report should state this in its protocol section, in one or two sentences.

## 8. Second-round review of the refactor

Three independent reviewers re-checked the refactored code, each followed by an adversarial verifier. Nothing
critical or high was found on the paths that produce the reported numbers. The confirmed items, all fixed before the
final run:

| Item | Fix |
|---|---|
| The notebook re-implemented the selection loop and overwrote `selection.json` with a reduced record | One `pipeline.select_models` for both; the notebook no longer writes into `reports/` |
| The test prevalence was printed before selection | `split.summary` hides it until the choices are frozen |
| The k-NN weights' inner split used a station rate that included the held-out week | The rate is recomputed on the inner fit part only |
| The population table omitted the 160 / 320 boundary drops | `split.population_table` (reason × block, totals match the modelling sets) |
| The 30-min ablation deltas were within gradient boosting's seed noise | Ablation with the random forest (seed std 0.0004) |
| The persistence score was not labelled; majority precision was shown as 0 | Labelled in the figure and README; precision is NaN when nothing is predicted |
| No per-station, weekday/weekend or matched-operating-point results | Added (`station_comparison`, `day_type`, `precision_at_recall`) |
| The pairing of incomplete snapshots was circular | Replaced by the count of distinct station sets |
| The paired bootstrap accepted misaligned frames | It now raises `ValueError` |
| Downloads could leave a truncated file; `assert`s were used for data checks | Atomic `.part` download with a timeout; explicit errors |
| Plot styling changed global matplotlib settings at import; the font was Windows-only | `rc_context` inside each function; DejaVu Sans fallback |
| The notebook generator was git-ignored and had no defaults or cell ids; the docstring and runtime claims were stale | Versioned under `tools/`, with defaults and ids; runtime measured (9.6–14.6 min) |
| The tests did not cover the inner purge, the selection protocol or the bootstrap | 6 tests added (16 in total) |

**A fix that was reverted.** To make the station rate "out of fold", training rows briefly received a
leave-one-week-out rate. A verifier showed that, within a station, this value is almost exactly the inverse of the
row's own-week empty rate (r = −0.999). Because the trees also identify the station from its coordinates, they can
decode the row's own-week labels, which flipped the 60-min model choice on validation. This is the classic
leave-one-out target-encoding leak. The station rate is again a single train-period constant (about 2,700 rows per
station, the same value for validation and test), and `test_station_rate_is_one_constant_per_station` guards it.

## 9. Caveats

* Results were recomputed with scikit-learn 1.9.1 / pandas 2.3.3, not in the original Deepnote environment.
* The random forest predicts with several threads, so its scores (and its F1 threshold) can differ in the 16th decimal
  between runs. No reported figure changes.
* The report and the slides themselves are not written: section 6 gives their structure and messages.
* Static gates: the code-quality scan of `src/velib` is clean (0 errors, 0 warnings). The security scan raises two
  accepted items, both in `data.py`.
  * "Possible SSRF" on `urlopen`: this is a pattern match. The URL is the hard-coded HTTPS constant
    `config.DATA_URL`, and `fetch` takes no URL argument.
  * "Weak hash MD5": the brief publishes an MD5 checksum, which is used here for integrity, not security.
