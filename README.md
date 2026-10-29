# Vélib' sous tension — Question A: *departure impossible*

M2 Data Science project (Université Paris-Saclay). Can we anticipate, **30 or 60 minutes ahead**, that a Vélib'
station will have **fewer than 2 bikes available**?

## Exact target

For station `s` and time `t` (15-minute UTC grid), with `h` ∈ {30, 60} minutes:

```
y(s, t) = 1  if  bikes_available(s, t + h) < 2   else 0
```

* Population: all 80 stations, one common model. A sample is kept when the station is **observed and open**
  (`is_renting = 1`) at `t` and at `t + h`, and `t + h` falls in the same block (train / validation / test) as `t`.
  Every excluded slot is counted by reason and by block (`reports/results/population_{30,60}min.csv`). Counting a
  closure at `t + h` as "departure impossible" instead is tested as a sensitivity analysis.
* References: the majority class, and **persistence**: `ŷ(s, t) = 1{bikes_available(s, t) < 2}`. For PR curves and
  PR-AUC, persistence ranks samples by the current number of bikes, a graded score that is generous to the reference.

## Data

| | |
|---|---|
| Source | Vélib' Métropole — real-time station availability, [Paris Data](https://opendata.paris.fr) |
| Licence | Open Database License (ODbL). Attribution: *Vélib' Métropole / Ville de Paris, Paris Data* |
| File | `velib-2026-10-08.csv.gz` — 329,680 rows, 80 stations (50 Paris, 30 suburbs), 25 Aug → 8 Oct 2026 |
| URL | https://docs.ambroise.cloud/velib/velib-2026-10-08.csv.gz |
| md5 | `02ab1e3fdac10579e4ce93d1540a5dc7` |

The file is **not versioned**. It is downloaded on first use into `data/`, and its md5 is checked before anything
else (`velib.data.fetch`). Manual alternative:

```bash
curl -o data/velib-2026-10-08.csv.gz https://docs.ambroise.cloud/velib/velib-2026-10-08.csv.gz
```

Timestamps are UTC; Paris is UTC+2 over the whole period. Calendar features are computed after
`tz_convert("Europe/Paris")`.

## Reproduce

```bash
python -m venv .venv
```

Activate it (`.venv\Scripts\activate` on Windows, `source .venv/bin/activate` elsewhere), then:

```bash
pip install -r requirements.txt
```

```bash
python tests/test_leakage_guards.py
```

```bash
python scripts/run_pipeline.py --extras
```

* The pipeline takes 10–15 minutes on a laptop (9.6 and 14.6 min measured, `--extras` included). It writes every table to
  `reports/results/` and every figure to `reports/figures/`, and `run_log.txt` keeps the printed output.
* `notebooks/velib_question_A.ipynb` is the commented walk-through. It calls the same functions and writes its figures
  to `notebooks/figures/` only. Open it in Jupyter or VS Code, or re-execute it headless with
  `python tools/execute_notebook.py notebooks/velib_question_A.ipynb`.
* `tools/build_notebook.py` regenerates the notebook from source; section 11 comes from `tools/conclusions.md`.

No installation is needed for the commands above. Optionally, `pip install -e .` (editable only) makes `import velib`
available everywhere.

## Protocol

| block | period (UTC) | role |
|---|---|---|
| train | 25 Aug 12:00 → 23 Sep 09:45 (start of the 28 h collection outage) | fitting |
| validation | 24 Sep 13:45 → 30 Sep 22:00 (= 1 Oct 00:00 Paris) | hyper-parameters, thresholds, model choice |
| test | 30 Sep 22:00 → 8 Oct 20:15 | scored once, after the choices are frozen in `reports/results/selection.json` |

* No random split anywhere. A sample is dropped if `t` and `t + h` fall in different blocks. The test block's labels
  (even its positive rate) are not printed before the choices are frozen.
* Lags (`d_bikes_15/30/60`, `bikes_24h`) are per-station shifts on a regular grid, read from open slots only. A
  missing or closed slot stays NaN; nothing is interpolated or forward-filled.
* Imputers and scalers live inside scikit-learn pipelines fitted on train. The station statistic `st_empty_rate`
  is measured on the training block only. The weighted k-NN's feature weights are learned inside the training block,
  with a purged inner hold-out.
* Metrics: PR-AUC, precision, recall, F1 and the confusion matrix. Results are broken down by area, hour band,
  weekday/weekend, day and station, with station-level bootstrap intervals. Accuracy is not reported: the useless
  majority class already scores 87–90 %.
* Disclosure: the first version of the project was scored on the test week during its audit (`reports/AUDIT.md`).
  The final code scores it once, after a choice made on validation by an unchanged rule.

## Layout

```
src/velib/      config · data · quality · grid · features · target · split · models · pipeline · evaluation · plots · experiments
scripts/        run_pipeline.py — end-to-end reproduction
tests/          test_leakage_guards.py — the brief's pitfalls and the protocol's leakage guards (16 tests)
notebooks/      velib_question_A.ipynb — narrative walk-through (executed), figures/
tools/          build_notebook.py, execute_notebook.py, conclusions.md — notebook generation
reports/        AUDIT.md, results/ (CSV, JSON, run log), figures/ (PNG)
```

## Licence

Code: MIT (see `LICENSE`). Data: ODbL, not redistributed here.
