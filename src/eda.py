"""Exploratory data analysis: audit report, figures, written limitations."""
from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import config, data_loader

log = logging.getLogger(__name__)

LIMITATIONS = [
    "Anonymous junction IDs (1-4): no coordinates, road names, lane counts or capacities are provided, so counts cannot be placed on a map.",
    "Vehicle COUNTS are not speeds, densities or travel times; congestion can only be expressed relatively.",
    "Junction 4 starts on 2017-01-01 (about 6 months) while Junctions 1-3 span Nov 2015 - Jun 2017: unequal history and unequal train size.",
    "About 20 months in total: yearly seasonality is barely observable; public holidays, weather and events are not recorded.",
    "No vehicle classes, direction of travel or sensor-fault flags; outliers (e.g. the Junction 3 maximum) cannot be verified as real or sensor errors.",
    "The data-collection city/country is not stated in the file; this project makes no geographic claim.",
]


def run(save: bool = True) -> dict:
    df = data_loader.load_raw()
    rep = data_loader.audit(df)
    rep["limitations"] = LIMITATIONS
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if save:
        data_loader.save_processed(df, rep)
    d = df.assign(hour=df.DateTime.dt.hour, dow=df.DateTime.dt.dayofweek, month=df.DateTime.dt.to_period("M").astype(str))

    fig, ax = plt.subplots(figsize=(12, 4))
    for j, g in d.groupby("Junction"):
        ax.plot(g.DateTime, g.Vehicles.rolling(24 * 7, min_periods=1).mean(), label=f"Junction {j}")
    ax.set(title="7-day rolling mean of hourly vehicle counts", ylabel="vehicles/hour"); ax.legend()
    fig.tight_layout(); fig.savefig(config.FIGURES_DIR / "trend_by_junction.png", dpi=120); plt.close(fig)

    for col, name, xl in (("hour", "hourly_profile", "hour of day"), ("dow", "weekday_profile", "day of week (0=Mon)")):
        fig, ax = plt.subplots(figsize=(8, 4))
        for j, g in d.groupby("Junction"):
            ax.plot(g.groupby(col).Vehicles.mean(), marker="o", label=f"Junction {j}")
        ax.set(title=f"Mean vehicles by {xl}", xlabel=xl, ylabel="mean vehicles/hour"); ax.legend()
        fig.tight_layout(); fig.savefig(config.FIGURES_DIR / f"{name}.png", dpi=120); plt.close(fig)

    m = d.groupby(["month", "Junction"]).Vehicles.mean().unstack()
    fig, ax = plt.subplots(figsize=(12, 4)); m.plot(ax=ax, marker="o")
    ax.set(title="Mean vehicles/hour by month", ylabel="vehicles/hour"); ax.tick_params(axis="x", rotation=60)
    fig.tight_layout(); fig.savefig(config.FIGURES_DIR / "monthly_mean.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for j, g in d.groupby("Junction"):
        ax.hist(g.Vehicles, bins=60, alpha=0.5, label=f"Junction {j}")
    ax.set(title="Distribution of hourly counts", xlabel="vehicles/hour"); ax.legend()
    fig.tight_layout(); fig.savefig(config.FIGURES_DIR / "distribution.png", dpi=120); plt.close(fig)

    piv = d.groupby(["dow", "hour"]).Vehicles.mean().unstack()
    fig, ax = plt.subplots(figsize=(10, 3.5)); im = ax.imshow(piv.values, aspect="auto", cmap="viridis")
    ax.set(title="Mean vehicles: weekday x hour (all junctions)", xlabel="hour", ylabel="day of week"); fig.colorbar(im)
    fig.tight_layout(); fig.savefig(config.FIGURES_DIR / "heatmap_dow_hour.png", dpi=120); plt.close(fig)
    rep["limitations"] = LIMITATIONS
    log.info("EDA complete: %d rows", rep["rows"])
    return rep


if __name__ == "__main__":
    import json
    config.setup_logging()
    print(json.dumps(run(), indent=2)[:3000])
