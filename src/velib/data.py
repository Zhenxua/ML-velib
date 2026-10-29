"""Download (cached), integrity check and loading of the frozen Vélib' snapshot file."""
import hashlib
import shutil
import urllib.request
from pathlib import Path

import pandas as pd

from .config import DATA_FILE, DATA_MD5, DATA_URL


def md5sum(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def fetch(path: Path = DATA_FILE, md5: str = DATA_MD5, timeout: int = 60) -> Path:
    """Download the published file (config.DATA_URL, fixed HTTPS address) only if it is missing, then always
    verify its md5 (brief: check before modelling). The download goes to a temporary '.part' file that is
    renamed only once complete, so an interrupted download never leaves a truncated file in place."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + ".part")
        with urllib.request.urlopen(DATA_URL, timeout=timeout) as response, open(part, "wb") as out:
            shutil.copyfileobj(response, out)
        part.replace(path)
    digest = md5sum(path)
    if digest != md5:
        raise ValueError(f"md5 mismatch for {path}: got {digest}, expected {md5}. "
                         "The file is not the published one: delete it and run again to re-download.")
    return path


def load_raw(path: Path = DATA_FILE) -> pd.DataFrame:
    """One row per (station, snapshot), sorted by station then collection time. Timestamps are tz-aware UTC."""
    path = fetch(path)
    df = pd.read_csv(path, parse_dates=["collected_at_utc", "source_updated_at"])
    for col in ("collected_at_utc", "source_updated_at"):
        if str(df[col].dt.tz) != "UTC":
            raise ValueError(f"{col} is not parsed as tz-aware UTC")
    return df.sort_values(["station_code", "collected_at_utc"]).reset_index(drop=True)
