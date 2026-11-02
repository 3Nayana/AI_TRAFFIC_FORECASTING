"""Leakage-safe feature engineering.

Conventions
-----------
* Everything lives on a complete regular hourly grid (missing hours = NaN), so a lag of k
  steps is *always* exactly k hours, never "k observations".
* A sample is defined by its ORIGIN time t. Features use data at times <= t only.
  The target is the count at t+h. Calendar features of the TARGET time are allowed
  because the calendar is known in advance.
* Samples are assigned to train/val/test by origin time, and every target
  (t+max_horizon) must fall inside the same split -> no target leaks across splits.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

LAGS = (0, 1, 2, 3, 6, 12, 24, 48, 168)


def calendar_features(index: pd.DatetimeIndex, hour_harm: int = 1, dow_harm: int = 1) -> pd.DataFrame:
    """Cyclical encodings of hour / weekday / month + weekend flag (no holidays available)."""
    out = {}
    hr = index.hour.values + index.minute.values / 60.0
    for k in range(1, hour_harm + 1):
        out[f"hour_sin{k}"] = np.sin(2 * np.pi * k * hr / 24)
        out[f"hour_cos{k}"] = np.cos(2 * np.pi * k * hr / 24)
    for k in range(1, dow_harm + 1):
        out[f"dow_sin{k}"] = np.sin(2 * np.pi * k * index.dayofweek.values / 7)
        out[f"dow_cos{k}"] = np.cos(2 * np.pi * k * index.dayofweek.values / 7)
    out["month_sin"] = np.sin(2 * np.pi * (index.month.values - 1) / 12)
    out["month_cos"] = np.cos(2 * np.pi * (index.month.values - 1) / 12)
    out["is_weekend"] = (index.dayofweek.values >= 5).astype(float)
    return pd.DataFrame(out, index=index)


def past_only_features(series: pd.Series) -> pd.DataFrame:
    """Lag and rolling statistics for each origin t using only values at times <= t.

    `series` must be on a regular grid (gaps = NaN). Rolling windows are strict
    (min_periods = window) so a gap makes the feature NaN instead of silently shrinking.
    """
    f = {f"lag_{k}": series.shift(k) for k in LAGS}
    f["roll_mean_24"] = series.rolling(24, min_periods=24).mean()
    f["roll_std_24"] = series.rolling(24, min_periods=24).std()
    f["roll_mean_168"] = series.rolling(168, min_periods=168).mean()
    return pd.DataFrame(f)


def tabular_frame(grid: pd.DataFrame, junction: int, horizon: int, use_calendar: bool = True) -> pd.DataFrame:
    """One row per origin time: past-only features + target-time calendar + y(t+h)."""
    s = grid[junction]
    X = past_only_features(s)
    if use_calendar:
        cal = calendar_features(grid.index + pd.Timedelta(hours=horizon), hour_harm=3, dow_harm=2)
        cal.index = grid.index  # row = origin; values describe the target time
        cal.columns = [f"tgt_{c}" for c in cal.columns]
        X = X.join(cal)
    X["y"] = s.shift(-horizon)
    X["target_time"] = grid.index + pd.Timedelta(hours=horizon)
    return X.dropna()


def in_split(origins: pd.DatetimeIndex, bounds: Tuple[pd.Timestamp, pd.Timestamp], max_h: int) -> np.ndarray:
    lo, hi = bounds
    return np.asarray((origins >= lo) & (origins + pd.Timedelta(hours=max_h) <= hi))


@dataclass
class Scaler:
    mean: Dict[int, float]
    std: Dict[int, float]

    @classmethod
    def fit(cls, grid: pd.DataFrame, train_end: pd.Timestamp) -> "Scaler":
        tr = grid.loc[:train_end]
        return cls({int(j): float(tr[j].mean()) for j in grid.columns},
                   {int(j): float(max(tr[j].std(), 1e-6)) for j in grid.columns})

    def fwd(self, j: int, v):
        return (v - self.mean[j]) / self.std[j]

    def inv(self, j: int, z):
        return z * self.std[j] + self.mean[j]


def sequence_samples(grid: pd.DataFrame, scaler: Scaler, bounds: Tuple[pd.Timestamp, pd.Timestamp],
                     window: int, horizons: Sequence[int], use_calendar: bool = True) -> Dict[str, np.ndarray]:
    """Windows for the LSTM. A window is kept only if all `window` past hours AND all target
    hours exist (no NaN) -> gaps are never bridged."""
    H = max(horizons)
    junctions = list(grid.columns)
    cal_all = calendar_features(grid.index).values.astype(np.float32)  # (T, 7)
    Xs, Es, Ys, Ts, Js, Rs = [], [], [], [], [], []
    for jn, j in enumerate(junctions):
        z = scaler.fwd(int(j), grid[j].values.astype(np.float64))
        T = len(z)
        if T < window + H:
            continue
        nan = np.isnan(z)
        win_nan = sliding_window_view(nan, window).any(axis=1)             # (T-window+1,)
        origins = np.arange(window - 1, T)
        ok = ~win_nan
        tgt_idx = origins[:, None] + np.asarray(horizons)[None, :]
        in_range = tgt_idx.max(axis=1) < T
        tg = np.full((len(origins), len(horizons)), np.nan)
        tg[in_range] = z[tgt_idx[in_range]]
        ok &= ~np.isnan(tg).any(axis=1)
        ok &= in_split(grid.index[origins], bounds, H)
        sel = np.where(ok)[0]
        if len(sel) == 0:
            continue
        o = origins[sel]
        win = sliding_window_view(z, window)[sel][..., None].astype(np.float32)   # (N, W, 1)
        if use_calendar:
            cwin = sliding_window_view(cal_all, window, axis=0)[sel].transpose(0, 2, 1)  # (N, W, 7)
            win = np.concatenate([win, cwin], axis=2)
        onehot = np.zeros((len(o), len(junctions)), np.float32); onehot[:, jn] = 1
        extra = [onehot]
        if use_calendar:
            extra += [cal_all[o + h] for h in horizons]   # target-time calendar, known in advance
        Xs.append(win); Es.append(np.concatenate(extra, axis=1)); Ys.append(tg[sel].astype(np.float32))
        Ts.append(grid.index[o].values); Js.append(np.full(len(o), int(j))); Rs.append(grid[j].values[o[:, None] + np.asarray(horizons)[None, :]])
    if not Xs:
        raise ValueError("No valid sequence samples for the requested split")
    return {"X": np.concatenate(Xs), "extra": np.concatenate(Es), "y": np.concatenate(Ys),
            "origin": np.concatenate(Ts), "junction": np.concatenate(Js), "y_raw": np.concatenate(Rs)}
