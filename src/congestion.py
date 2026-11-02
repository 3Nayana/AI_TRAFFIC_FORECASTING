"""Relative congestion classes from historical vehicle COUNTS.

The classes are *relative to each junction's own TRAIN-period distribution*
(default quantiles 50/75/90 %). They say "busier than usual for this junction",
NOT that traffic is slow: counts are not speeds, and a saturated road can show LOW counts.
Counts are never converted to travel times here.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd

from . import config


def fit_thresholds(grid: pd.DataFrame, train_end: pd.Timestamp,
                   quantiles: Sequence[float] = config.CONGESTION_QUANTILES,
                   by_hour: bool = False) -> Dict[int, Dict]:
    """Per-junction thresholds from training data only. `by_hour=True` computes them per hour of day
    (accounts for the daily cycle; needs more data per bucket)."""
    tr = grid.loc[:train_end]
    out: Dict[int, Dict] = {}
    for j in grid.columns:
        s = tr[j].dropna()
        if by_hour:
            out[int(j)] = {int(h): [float(x) for x in g.quantile(list(quantiles))]
                           for h, g in s.groupby(s.index.hour)}
        else:
            out[int(j)] = {"all": [float(x) for x in s.quantile(list(quantiles))]}
    return out


def classify(values, junction: int, hours, thresholds: Dict[int, Dict],
             labels: Sequence[str] = config.CONGESTION_LABELS) -> np.ndarray:
    values = np.asarray(values, float); hours = np.asarray(hours)
    th = thresholds[int(junction)]
    out = np.empty(len(values), dtype=object)
    for i, (v, h) in enumerate(zip(values, hours)):
        t = th["all"] if "all" in th else th[int(h)]
        out[i] = labels[int(np.searchsorted(t, v, side="right"))]
    return out


LIMITATIONS = ("Thresholds are statistical, per junction, from the training period only. A class says how busy the "
               "junction is relative to its own history; it does not measure speed, delay or capacity utilisation.")
