"""Features available at prediction time t (past and present only).

Rules enforced here:
  - every lag is a per-station shift on the regular grid (groupby station_code), so a lag never
    reads another station's row and a missing slot gives NaN instead of an older value;
  - calendar features are computed in Paris local time (tz_convert), not in UTC;
  - the only statistic estimated from data (st_empty_rate) is computed on the training period only.
"""
import numpy as np
import pandas as pd

from .config import EMPTY_THRESHOLD, LOCAL_TZ, STEP_MIN, T_VAL

FEATURE_GROUPS = {
    "state": ["bikes_available", "docks_available", "fill"],
    "trend": ["d_bikes_15", "d_bikes_30", "d_bikes_60", "bikes_24h"],
    "calendar": ["hour_sin", "hour_cos", "dow", "weekend"],
    "station": ["capacity", "paris", "latitude", "longitude", "st_empty_rate"],
}
FEATURES = [f for group in FEATURE_GROUPS.values() for f in group]
# Distance-based (k-NN) and Gaussian (naive Bayes) models would read `dow` (0..6) as an ordered
# quantity — Sunday "far" from Monday. They use `weekend` + hour of day instead.
FEATURES_NO_DOW = [f for f in FEATURES if f != "dow"]


def add_features(g: pd.DataFrame, train_end: pd.Timestamp = T_VAL) -> pd.DataFrame:
    g = g.copy()

    # 1. Current state. `fill` uses bikes + docks rather than the declared capacity, which is
    #    inconsistent for a few stations (bikes + docks > capacity); 0/0 -> NaN.
    total = g["bikes_available"] + g["docks_available"]
    g["fill"] = g["bikes_available"] / total.where(total > 0)

    # 2. Recent trend and daily lag, per station: shift(k) = exactly k x 15 min earlier. Lags are read
    #    from open slots only: a closed station mostly reports 0 bikes, which is not an availability
    #    state, so a lag pointing into a closure is NaN — exactly like a lag pointing into an outage.
    bikes_open = g["bikes_available"].where(~g["closed"]).groupby(level="station_code", sort=False)
    for k in (1, 2, 4):
        g[f"d_bikes_{STEP_MIN * k}"] = g["bikes_available"] - bikes_open.shift(k)
    g["bikes_24h"] = bikes_open.shift(24 * 60 // STEP_MIN)

    # 3. Calendar, in Paris local time (UTC+2 here; tz_convert also handles DST if the data is extended).
    t_local = g.index.get_level_values("t").tz_convert(LOCAL_TZ)
    hour = t_local.hour + t_local.minute / 60
    g["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    g["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    g["dow"] = t_local.dayofweek
    g["weekend"] = (g["dow"] >= 5).astype(int)

    # 4. Station attributes. st_empty_rate = share of open, observed slots with < 2 bikes, measured on the
    #    training block only and applied as the same constant to every row (train, validation, test).
    #    Deliberately NOT out of fold: a leave-one-week-out rate is, within a station, an almost exact inverse
    #    of that week's own empty rate, and since the trees also identify the station (latitude, longitude,
    #    capacity) they would decode the row's own-week labels — a leak that does not transfer to new weeks.
    #    A full-train constant averages ~2,700 rows per station, so each row's own contribution is negligible.
    g["paris"] = g["commune"].eq("Paris").astype(int)
    in_train = g.index.get_level_values("t") < train_end
    empty_now = (g["bikes_available"] < EMPTY_THRESHOLD).astype(float).where(g["observed"] & ~g["closed"])
    rate = empty_now[in_train].groupby(level="station_code").mean()
    g["st_empty_rate"] = rate.reindex(g.index.get_level_values("station_code")).to_numpy()
    return g
