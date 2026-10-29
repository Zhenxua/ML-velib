"""Target of Question A: "departure impossible" h minutes ahead."""
import pandas as pd

from .config import EMPTY_THRESHOLD


def make_target(g: pd.DataFrame, steps: int, closed_as_positive: bool = False) -> tuple[pd.Series, pd.Series]:
    """y(station, t) = 1 if the station has fewer than EMPTY_THRESHOLD bikes at t + steps x 15 min.

    The future value is taken with a per-station shift(-steps) on the regular grid, so it is the same
    station exactly h minutes later (never the next station, never "the next available snapshot").

    Population (returned as the boolean mask `valid`):
      - observed and open at t: the prediction is made from an observed state of a working station;
      - observed at t + h, and open at t + h. A closure (is_renting == 0) is a different phenomenon
        from running out of bikes, so by default those samples are excluded — and *counted* in the
        report. With closed_as_positive=True a closure at t + h counts as "departure impossible"
        (y = 1) instead: this is the sensitivity analysis of that choice.
    """
    by_station = g.groupby(level="station_code", sort=False)
    future_bikes = by_station["bikes_available"].shift(-steps)
    future_observed = by_station["observed"].shift(-steps, fill_value=False)
    future_closed = by_station["closed"].shift(-steps, fill_value=False)

    y = future_bikes < EMPTY_THRESHOLD
    valid = g["observed"] & ~g["closed"] & future_observed
    if closed_as_positive:
        y = y | future_closed
    else:
        valid &= ~future_closed
    return y[valid].astype(int), valid
