"""Speed -> travel-time maths and the vocabulary for *where a speed came from*.

travel_time_seconds = length_m / (speed_kmh * 1000 / 3600)
"""
from __future__ import annotations

import math
from enum import Enum
from typing import Optional

MIN_SPEED_KMH = 5.0       # floor for *open* edges (closures are handled separately, never as speed ~0)
MAX_SPEED_KMH = 130.0
DEFAULT_FALLBACK_KMH = 30.0


class SpeedKind(str, Enum):
    SPEED_LIMIT = "speed_limit"            # legal/tagged limit
    FREE_FLOW_ASSUMED = "free_flow_assumed"  # assumed baseline = factor * limit (not measured)
    SCENARIO_SYNTHETIC = "scenario_synthetic"  # produced by a demo scenario (NOT observed, NOT predicted)
    OBSERVED = "observed"                  # measured by an authorised source
    PREDICTED = "predicted"                # forecast, geographically matched to the edge


def kmh_to_ms(speed_kmh: float) -> float:
    return speed_kmh * 1000.0 / 3600.0


def is_valid_speed(speed: Optional[float]) -> bool:
    return speed is not None and isinstance(speed, (int, float)) and math.isfinite(speed) and speed > 0


def sanitize_speed(speed: Optional[float], fallback: float = DEFAULT_FALLBACK_KMH) -> tuple[float, bool]:
    """Return (speed, used_fallback). Clamps into [MIN, MAX]; invalid -> fallback."""
    if not is_valid_speed(speed):
        return float(min(max(fallback, MIN_SPEED_KMH), MAX_SPEED_KMH)), True
    return float(min(max(speed, MIN_SPEED_KMH), MAX_SPEED_KMH)), False


def travel_time_seconds(length_m: float, speed_kmh: float) -> float:
    if not (math.isfinite(length_m) and length_m >= 0):
        raise ValueError(f"invalid length_m={length_m}")
    if not is_valid_speed(speed_kmh):
        raise ValueError(f"speed must be positive and finite, got {speed_kmh}")
    return length_m / kmh_to_ms(speed_kmh)


def congestion_index(speed_kmh: float, freeflow_kmh: float) -> float:
    """0 = free flow, 1 = extremely slow. c = clip(1 - v/v_ff, 0, 1)."""
    if not is_valid_speed(freeflow_kmh):
        return 0.0
    return float(min(max(1.0 - speed_kmh / freeflow_kmh, 0.0), 1.0))


LEVELS = ("free", "light", "moderate", "heavy")
LEVEL_COLORS = {"free": "#2ca02c", "light": "#bcbd22", "moderate": "#ff7f0e", "heavy": "#d62728", "closed": "#000000"}


def congestion_level(c: float, edges=(0.2, 0.4, 0.6)) -> str:
    """Speed-ratio based levels: ratio=v/v_ff >0.8 free, 0.6-0.8 light, 0.4-0.6 moderate, <0.4 heavy."""
    return LEVELS[sum(c >= e for e in edges)]
