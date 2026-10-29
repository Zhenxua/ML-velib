*Numbers below are those produced by this notebook and by `scripts/run_pipeline.py` (fixed seeds; the random forest's
multi-threaded predictions can differ in the 16th decimal between runs, which does not change any reported figure).*

**Answer to the question.** On the test week (1–8 October):

| horizon | model chosen on validation | PR-AUC model / persistence | F1 model / persistence | difference, 95 % station-bootstrap interval |
|---|---|---|---|---|
| 30 min | random forest | 0.816 / 0.702 | 0.742 / 0.729 | PR-AUC +0.114 [0.101; 0.128] · F1 +0.013 [0.007; 0.020] |
| 60 min | gradient boosting | 0.754 / 0.619 | 0.689 / 0.660 | PR-AUC +0.134 [0.118; 0.150] · F1 +0.030 [0.019; 0.040] |

The model mainly improves the **ranking** of stations by risk (PR-AUC). At persistence's own operating point the gain
is smaller: at the same recall, precision goes from 0.729 to 0.749 at 30 minutes and from 0.660 to 0.700 at 60
minutes. Persistence's PR-AUC is computed with a graded score (fewer bikes now = higher risk), which is generous to the
reference.

**Characteristic errors.**
* **Where the gain comes from.** Persistence is right whenever the state does not change, which covers 93 % of the test
  samples at 30 minutes. The model's value lies in the transitions. It anticipates about 21 % of the stations that
  *become* empty within 30 minutes (32 % at 60 minutes), which persistence misses by construction. It still predicts
  "empty" for 95 % of the empty stations that recover (persistence: 100 %).
* **By hour (Paris time).** Nights are easy (PR-AUC 0.92) and give no F1 gain over persistence. The evening peak,
  17–20 h, is the hardest band (PR-AUC 0.69).
* **By day.** Weekends are harder than weekdays (PR-AUC 0.78 vs 0.83). Day to day, the test F1 is 0.741 ± 0.032; it is
  lowest on Sunday 4 October, which is also the worst day for persistence.
* **By area and station.** Paris and suburban stations reach similar scores. Per station, the model's F1 is at least
  that of persistence on 68 % of the stations that have a positive on the test week (59 % at 60 minutes). The
  largest losses are on stations with very few positives (4 to 30 in the week).

**What the features bring** (validation, random forest refitted without each group).
* The current state is essential: −0.27 PR-AUC without it.
* The calendar group (mostly the hour of day) comes next: −0.02 at 30 minutes, −0.04 at 60 minutes.
* Station attributes add a little (−0.01 / −0.015), and so do recent trends (−0.01 at 30 minutes, nothing at 60).
* A model restricted to the current state (bikes, docks, fill ratio) already ranks better than persistence:
  validation PR-AUC 0.757 vs 0.707 at 30 minutes.

These are associations measured on this period, not effects.

**Model choice.** The random forest and gradient boosting are tied: over five seeds, gradient boosting's validation
PR-AUC varies by ±0.008 at 30 minutes, more than the gap between the two. Choosing on the five-seed mean selects the
same models. The depth-8 tree and the importance-weighted k-NN follow closely (test PR-AUC ≈ 0.80 at 30 minutes).
Plain k-NN and naive Bayes do not beat persistence on PR-AUC.

**Robustness.** Counting a closure at t + h as "departure impossible" adds 5 / 8 test samples and changes no metric at
three decimals. Removing station 45504, which may be frozen in the feed, changes nothing either: it has no positive.

**Limits of the collection.**
* Six weeks and 80 stations, in a single season (end of summer to beginning of autumn 2026).
* A 28-hour outage just before the validation week, so the 24-hour lag is missing on its first day.
* 36 incomplete snapshots, in which the same block of 30 or 50 stations is missing.
* Four stations closed for one to two days.
* One station (45504) that may be frozen in the feed.
* A source timestamp that is not a reliable freshness signal.
* A positive rate drifting upward week after week (10.3 % in train, 12.2 % in test).
* No weather, events or rebalancing information.

**Protocol caveat.** All choices were made on validation by fixed rules, and the test week was scored once by the final
code. However, during the audit of the first version of this project, the frozen first-version models were also
scored on the test week. The changes made afterwards each follow from the brief's rules or from training/validation
evidence, and the selection rule (best validation PR-AUC) is unchanged. They are listed in `reports/AUDIT.md`.

**Limits of use.** The model predicts an *observed state of the service* — the counts published by the feed — 30 or
60 minutes ahead, for these 80 stations, under the conditions of the collection period. It does not explain why a
station empties: departures, missing rebalancing and data errors cannot be told apart here. It is not a model of
individual users' demand. Any use on other stations, another season or another year would require a new evaluation.
