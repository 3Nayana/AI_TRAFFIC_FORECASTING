"""Central, location-independent configuration."""
from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "traffic.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
DEMO_DIR = ROOT / "data" / "demo"
ROAD_CACHE_DIR = ROOT / "data" / "road_cache"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

HORIZONS = (1, 3, 6, 24)          # hours ahead (dataset is hourly)
SEEDS = (0, 1, 2)                 # LSTM seeds
LSTM_WINDOW = 48                  # hours of history fed to the LSTM

# Chronological split by *origin time* (inclusive end dates). Computed from data in
# data_loader.default_split(); these are the defaults used for the Kaggle file.
TRAIN_END = "2017-02-28 23:00"
VAL_END = "2017-04-30 23:00"
TEST_END = "2017-06-30 23:00"

# Relative congestion quantiles (of TRAIN-period counts, per junction)
CONGESTION_QUANTILES = (0.50, 0.75, 0.90)
CONGESTION_LABELS = ("Low", "Moderate", "High", "Very high")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:  # pragma: no cover
        pass


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)
