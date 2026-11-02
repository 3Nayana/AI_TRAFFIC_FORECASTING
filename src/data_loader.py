"""Load, validate and audit the Kaggle traffic.csv (columns DateTime, Junction, Vehicles, ID)."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from . import config

log = logging.getLogger(__name__)
REQUIRED_COLUMNS = ["DateTime", "Junction", "Vehicles", "ID"]


class DataValidationError(ValueError):
    """Raised when the dataset does not match the expected schema."""


def load_raw(path: Path | str = config.RAW_CSV) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download traffic.csv from "
            "https://www.kaggle.com/datasets/fedesoriano/traffic-prediction-dataset "
            "and place it at data/raw/traffic.csv")
    df = pd.read_csv(path)
    return validate(df)


def validate(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")
    out = df[REQUIRED_COLUMNS].copy()
    out["DateTime"] = pd.to_datetime(out["DateTime"], errors="coerce")
    if out["DateTime"].isna().any():
        raise DataValidationError(f"{int(out['DateTime'].isna().sum())} unparsable DateTime values")
    for c in ("Junction", "Vehicles"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
        if out[c].isna().any():
            raise DataValidationError(f"Column {c} has non-numeric values")
    if (out["Vehicles"] < 0).any():
        raise DataValidationError("Negative vehicle counts found")
    out["Junction"] = out["Junction"].astype(int)
    out["Vehicles"] = out["Vehicles"].astype(float)
    return out.sort_values(["Junction", "DateTime"]).reset_index(drop=True)


def audit(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute facts from the file; nothing is assumed."""
    rep: Dict[str, Any] = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "missing_values": {c: int(v) for c, v in df.isna().sum().items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_junction_timestamp": int(df.duplicated(["DateTime", "Junction"]).sum()),
        "start": str(df["DateTime"].min()),
        "end": str(df["DateTime"].max()),
        "junctions": {},
    }
    for j, g in df.groupby("Junction"):
        g = g.sort_values("DateTime")
        d = g["DateTime"].drop_duplicates().diff().dropna()   # duplicates must not count as observed hours
        mode = d.mode().iloc[0] if len(d) else pd.Timedelta(0)
        expected = int((g["DateTime"].max() - g["DateTime"].min()) / mode) + 1 if mode > pd.Timedelta(0) else len(g)
        gaps = d[d > mode]
        rep["junctions"][int(j)] = {
            "rows": int(len(g)),
            "start": str(g["DateTime"].min()),
            "end": str(g["DateTime"].max()),
            "modal_interval": str(mode),
            "interval_counts": {str(k): int(v) for k, v in d.value_counts().head(5).items()},
            "expected_rows_if_complete": expected,
            "missing_timestamps": int(expected - g["DateTime"].nunique()),
            "n_gaps": int(len(gaps)),
            "largest_gap": str(gaps.max()) if len(gaps) else None,
            "vehicles": {k: float(v) for k, v in g["Vehicles"].describe().round(3).items()},
        }
    rep["id_column_note"] = "ID is a concatenation key (date + junction); not used as a feature."
    return rep


def to_hourly_grid(df: pd.DataFrame, freq: str = "h") -> pd.DataFrame:
    """Wide table on a *complete* regular time grid; missing hours become NaN.

    Because gaps stay NaN, shift/rolling features can never silently treat
    non-consecutive observations as consecutive.
    """
    wide = df.pivot(index="DateTime", columns="Junction", values="Vehicles").sort_index()
    full = pd.date_range(wide.index.min(), wide.index.max(), freq=freq)
    wide = wide.reindex(full)
    wide.index.name = "DateTime"
    return wide


def save_processed(df: pd.DataFrame, audit_report: Dict[str, Any]) -> None:
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.PROCESSED_DIR / "traffic_clean.csv", index=False)
    to_hourly_grid(df).to_csv(config.PROCESSED_DIR / "traffic_hourly_grid.csv")
    (config.PROCESSED_DIR / "audit.json").write_text(json.dumps(audit_report, indent=2))


def split_bounds() -> Dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """Chronological train/val/test bounds on origin time (inclusive)."""
    tr_end, va_end, te_end = (pd.Timestamp(x) for x in (config.TRAIN_END, config.VAL_END, config.TEST_END))
    return {
        "train": (pd.Timestamp.min, tr_end),
        "val": (tr_end + pd.Timedelta(hours=1), va_end),
        "test": (va_end + pd.Timedelta(hours=1), te_end),
    }
