"""Project-wide constants: data source, question definition, split boundaries, seed.

Everything that defines the protocol lives here so that it is written down once and
cannot drift between the notebook, the script and the tests.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "reports" / "results"
FIGURES_DIR = ROOT / "reports" / "figures"

# --- Frozen dataset (not versioned: downloaded once, then checked against its md5) ---
DATA_URL = "https://docs.ambroise.cloud/velib/velib-2026-10-08.csv.gz"
DATA_FILE = DATA_DIR / "velib-2026-10-08.csv.gz"
DATA_MD5 = "02ab1e3fdac10579e4ce93d1540a5dc7"

# --- Time ---
# Timestamps are UTC. Paris is UTC+2 over the whole collection (DST ends 2026-10-25);
# local-time features always go through tz_convert(LOCAL_TZ), never a hard-coded +2.
LOCAL_TZ = "Europe/Paris"
FREQ = "15min"
STEP_MIN = 15
MAX_ALIGN_OFFSET = pd.Timedelta("1min")  # snapshots further than this from a quarter hour are dropped

# --- Question A: "departure impossible" ---
# y(station, t) = 1  <=>  bikes_available(station, t + h) < EMPTY_THRESHOLD
EMPTY_THRESHOLD = 2
HORIZONS = {30: 2, 60: 4}  # horizon in minutes -> number of 15-min steps

# --- Chronological split (all boundaries in UTC) ---
# train: 2026-08-25 12:00 -> the 28 h collection outage (last slot 2026-09-23 09:45 UTC)
# val:   first slot after the outage (2026-09-24 13:45 UTC) -> 2026-10-01 00:00 Paris
# test:  2026-10-01 00:00 Paris -> end of collection (2026-10-08 20:15 UTC), untouched until the model is chosen
T_VAL = pd.Timestamp("2026-09-24 13:45", tz="UTC")
T_TEST = pd.Timestamp("2026-10-01 00:00", tz=LOCAL_TZ).tz_convert("UTC")
SPLITS = ("train", "val", "test")

SEED = 0
