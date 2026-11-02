"""Metrics, model comparison tables and forecast plots (all numbers come from saved predictions)."""
from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config

log = logging.getLogger(__name__)


def mae(y, p): return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))
def rmse(y, p): return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))
def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ss = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum((y - p) ** 2) / ss) if ss > 0 else float("nan")


def metrics_table(pred: pd.DataFrame) -> pd.DataFrame:
    """pred columns: model, junction, horizon, actual, pred (+ origin_time, target_time)."""
    rows = []
    for (m, j, h), g in pred.groupby(["model", "junction", "horizon"]):
        rows.append(dict(model=m, junction=j, horizon=h, n=len(g), MAE=mae(g.actual, g.pred),
                         RMSE=rmse(g.actual, g.pred), R2=r2(g.actual, g.pred)))
    return pd.DataFrame(rows)


def add_skill(metrics: pd.DataFrame, baseline: str) -> pd.DataFrame:
    b = metrics[metrics.model == baseline][["junction", "horizon", "MAE"]].rename(columns={"MAE": "MAE_base"})
    out = metrics.merge(b, on=["junction", "horizon"], how="left")
    out["MAE_skill_vs_" + baseline] = 1 - out["MAE"] / out["MAE_base"]
    return out.drop(columns="MAE_base")


def plot_actual_vs_pred(pred: pd.DataFrame, junction: int, horizon: int, models, path, days: int = 7) -> None:
    g = pred[(pred.junction == junction) & (pred.horizon == horizon)]
    fig, ax = plt.subplots(figsize=(12, 4))
    base = g[g.model == models[0]].sort_values("target_time")
    t0 = base.target_time.min()
    base = base[base.target_time < t0 + pd.Timedelta(days=days)]
    ax.plot(base.target_time, base.actual, "k", lw=1.5, label="actual")
    for m in models:
        gm = g[(g.model == m)].sort_values("target_time")
        gm = gm[gm.target_time < t0 + pd.Timedelta(days=days)]
        ax.plot(gm.target_time, gm.pred, lw=1, label=m)
    ax.set(title=f"Junction {junction}, horizon +{horizon}h (test period, first {days} days)", ylabel="vehicles/hour")
    ax.legend(ncol=4, fontsize=8); fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
