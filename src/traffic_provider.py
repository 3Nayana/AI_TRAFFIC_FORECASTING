"""Traffic-speed provider interface (the documented bridge between Module A and Module B).

Module B only needs, per directed road edge, a validated speed (km/h), the *kind* of that speed and a timestamp.
Providers:
  FreeFlowProvider      assumed free-flow speeds (no traffic information)
  ScenarioProvider      SYNTHETIC demo scenarios on the demo graph (never observed / predicted)
  EdgeSpeedFileProvider geographically matched OBSERVED or PREDICTED edge speeds from a CSV
                        (columns: edge_id, timestamp, speed_kmh) with freshness + coverage validation
  LiveAPIProvider       skeleton for an authorised API (keys from env vars); never claims "live" without fresh data

The Kaggle junction counts are NOT a provider: they carry no geography. To use Module A's forecasts here you
must first produce *edge-level* speeds from a geographically matched source and feed them via EdgeSpeedFileProvider.
"""
from __future__ import annotations

import logging
import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import networkx as nx
import numpy as np
import pandas as pd

from .demo_network import scenario_multipliers, scenario_speed, SCENARIOS
from .travel_time import SpeedKind, sanitize_speed

log = logging.getLogger(__name__)


@dataclass
class EdgeSpeed:
    speed_kmh: float
    kind: SpeedKind
    closed: bool = False
    timestamp: Optional[datetime] = None


@dataclass
class ProviderStatus:
    kind: SpeedKind
    label: str
    usable: bool
    coverage: float = 1.0
    synthetic: bool = False
    forecast_based: bool = False
    time_dependent: bool = False
    newest_timestamp: Optional[datetime] = None
    warnings: List[str] = field(default_factory=list)


class TrafficProvider(ABC):
    name = "provider"
    is_time_dependent = False

    @abstractmethod
    def speeds(self, G: nx.DiGraph, when: datetime) -> Dict[str, EdgeSpeed]: ...

    @abstractmethod
    def status(self, G: nx.DiGraph, when: datetime) -> ProviderStatus: ...


class FreeFlowProvider(TrafficProvider):
    name = "Assumed free-flow (no traffic data)"

    def speeds(self, G, when):
        return {d["edge_id"]: EdgeSpeed(d["freeflow_kmh"], SpeedKind.FREE_FLOW_ASSUMED, False, when) for _, _, d in G.edges(data=True)}

    def status(self, G, when):
        return ProviderStatus(SpeedKind.FREE_FLOW_ASSUMED, "STATIC FREE-FLOW (assumed, no traffic information)", True,
                              warnings=["No traffic information: congestion terms are zero and routes ignore current conditions."])


class ScenarioProvider(TrafficProvider):
    """SYNTHETIC scenario speeds for the demonstration graph. Illustrative only."""

    def __init__(self, scenario: str = "normal", severity: float = 0.7):
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario {scenario!r}")
        self.scenario, self.severity = scenario, float(min(max(severity, 0), 1))
        self.name = f"Synthetic scenario: {SCENARIOS[scenario]} (severity {self.severity:.2f})"
        self.is_time_dependent = scenario == "daily_cycle"

    def speeds(self, G, when):
        hour = when.hour + when.minute / 60 if self.is_time_dependent else None
        mult = scenario_multipliers(G, self.scenario, self.severity, hour)
        out = {}
        for _, _, d in G.edges(data=True):
            m, closed = mult[d["edge_id"]]
            out[d["edge_id"]] = EdgeSpeed(scenario_speed(d["freeflow_kmh"], m), SpeedKind.SCENARIO_SYNTHETIC, closed, when)
        return out

    def status(self, G, when):
        w = ["SYNTHETIC speeds: illustrative, not real traffic, not observed and not a forecast from the Kaggle-trained model."]
        return ProviderStatus(SpeedKind.SCENARIO_SYNTHETIC, "SYNTHETIC SCENARIO", True, synthetic=True,
                              time_dependent=self.is_time_dependent, warnings=w)


class EdgeSpeedFileProvider(TrafficProvider):
    """OBSERVED or PREDICTED speeds already matched to edge_ids of the loaded graph."""

    REQUIRED = ("edge_id", "timestamp", "speed_kmh")

    def __init__(self, df: pd.DataFrame, kind: SpeedKind = SpeedKind.PREDICTED, max_age_minutes: float = 15,
                 forecast_tolerance_minutes: float = 30, min_coverage: float = 0.5, speed_unit: str = "kmh",
                 closed_col: str = "closed"):
        if kind not in (SpeedKind.OBSERVED, SpeedKind.PREDICTED):
            raise ValueError("kind must be OBSERVED or PREDICTED")
        miss = [c for c in self.REQUIRED if c not in df.columns]
        if miss:
            raise ValueError(f"missing columns {miss}; required {list(self.REQUIRED)}")
        factor = {"kmh": 1.0, "mph": 1.609344, "ms": 3.6}.get(speed_unit)
        if factor is None:
            raise ValueError("speed_unit must be one of kmh, mph, ms")
        d = df.copy()
        d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
        d["speed_kmh"] = pd.to_numeric(d["speed_kmh"], errors="coerce") * factor
        d["edge_id"] = d["edge_id"].astype(str)
        d["closed"] = d[closed_col].astype(bool) if closed_col in d.columns else False
        self.n_bad_rows = int((d.timestamp.isna() | ~np.isfinite(d.speed_kmh) | (d.speed_kmh <= 0)).sum())
        self.df = d[~(d.timestamp.isna())].sort_values("timestamp")
        self.kind, self.max_age, self.tol, self.min_cov = kind, max_age_minutes, forecast_tolerance_minutes, min_coverage
        self.is_time_dependent = self.df.timestamp.nunique() > 1
        self.name = f"{kind.value} edge speeds from file"

    def _select(self, when: datetime) -> pd.DataFrame:
        d = self.df
        if self.kind == SpeedKind.OBSERVED:      # latest observation not newer than `when` and not too old
            d = d[(d.timestamp <= pd.Timestamp(when)) & (d.timestamp >= pd.Timestamp(when) - pd.Timedelta(minutes=self.max_age))]
            return d.groupby("edge_id").tail(1)
        # PREDICTED: closest forecast timestamp within tolerance
        d = d.assign(dt=(d.timestamp - pd.Timestamp(when)).abs())
        d = d[d.dt <= pd.Timedelta(minutes=self.tol)]
        return d.sort_values("dt").groupby("edge_id").head(1)

    def speeds(self, G, when):
        sel = self._select(when); ids = {d["edge_id"] for _, _, d in G.edges(data=True)}
        out = {}
        for r in sel.itertuples():
            if r.edge_id in ids:
                if r.closed:
                    out[r.edge_id] = EdgeSpeed(5.0, self.kind, True, r.timestamp.to_pydatetime())
                elif math.isfinite(r.speed_kmh) and r.speed_kmh > 0:
                    out[r.edge_id] = EdgeSpeed(sanitize_speed(r.speed_kmh)[0], self.kind, False, r.timestamp.to_pydatetime())
        return out   # missing edges are filled (and flagged) by the router

    def status(self, G, when):
        sel = self.speeds(G, when); n = G.number_of_edges()
        cov = len(sel) / n if n else 0.0
        w, usable = [], True
        if self.n_bad_rows:
            w.append(f"{self.n_bad_rows} rows ignored (bad timestamp or non-positive/NaN speed).")
        unknown = set(self.df.edge_id) - {d["edge_id"] for _, _, d in G.edges(data=True)}
        if unknown:
            w.append(f"{len(unknown)} edge_ids in the file do not exist in the loaded road graph (not geographically matched).")
        if not sel:
            usable = False
            w.append("No fresh/valid records for the requested time (stale or outside the forecast tolerance): using free-flow speeds instead.")
        elif cov < self.min_cov:
            usable = False
            w.append(f"Coverage {cov:.0%} is below the required {self.min_cov:.0%}: traffic data not used.")
        elif cov < 1:
            w.append(f"Only {cov:.0%} of edges have data; the rest use assumed free-flow speeds.")
        newest = max((s.timestamp for s in sel.values() if s.timestamp), default=None)
        return ProviderStatus(self.kind, self.kind.value.upper() + " (from file)", usable, cov, False,
                              self.kind == SpeedKind.PREDICTED, self.is_time_dependent, newest, w)


class LiveAPIProvider(TrafficProvider):
    """Skeleton for an *authorised* traffic API. No vendor is hard-wired.

    Implement `fetch()` to return a DataFrame with columns edge_id, timestamp, speed_kmh (already matched to the
    graph), obeying the vendor's terms (rate limits, caching, attribution). Credentials come from the environment.
    """
    name = "Live traffic API (not configured)"

    def __init__(self):
        self.key = os.environ.get("TRAFFIC_API_KEY", "")
        self.max_age = float(os.environ.get("TRAFFIC_API_MAX_AGE_MINUTES", "15") or 15)

    def fetch(self) -> pd.DataFrame:  # pragma: no cover - vendor specific
        raise NotImplementedError("Implement a vendor adapter returning edge_id, timestamp, speed_kmh")

    def speeds(self, G, when):
        return {}

    def status(self, G, when):
        w = ["Live traffic is NOT available: no vendor adapter / credentials configured. Showing offline mode only."]
        if not self.key:
            w.append("TRAFFIC_API_KEY is not set (see .env.example).")
        return ProviderStatus(SpeedKind.OBSERVED, "LIVE API (unavailable)", False, 0.0, warnings=w)


# ------------------------------------------------------------------ geographic matching of raw observations
def _local_xy(lat, lon, lat0, lon0):
    k = 111_320.0
    return (np.asarray(lon) - lon0) * math.cos(math.radians(lat0)) * k, (np.asarray(lat) - lat0) * k


def _bearing(x1, y1, x2, y2):
    return (np.degrees(np.arctan2(x2 - x1, y2 - y1)) + 360) % 360


def match_observations_to_edges(G: nx.DiGraph, obs: pd.DataFrame, max_dist_m: float = 30.0,
                                ambiguity_m: float = 8.0, heading_tol_deg: float = 45.0) -> pd.DataFrame:
    """Map point observations (lat, lon, speed_kmh, timestamp[, heading]) to directed edges.

    Defensible rules: (1) nearest edge geometry within `max_dist_m`; (2) if several edges are within `ambiguity_m` of
    the best, a heading is required to choose the direction/road, otherwise the point is rejected as ambiguous
    (a two-way street has two directed edges and a point alone cannot tell which one it measured).
    One-way edges with no competing candidates are accepted without heading.
    Returns the input with columns edge_id (NaN if rejected) and match_status.
    """
    lat0 = float(np.mean([d["y"] for _, d in G.nodes(data=True)])); lon0 = float(np.mean([d["x"] for _, d in G.nodes(data=True)]))
    segs = []
    for u, v, d in G.edges(data=True):
        pts = d.get("geometry") or [(G.nodes[u]["y"], G.nodes[u]["x"]), (G.nodes[v]["y"], G.nodes[v]["x"])]
        for (la, lo), (lb, lb2) in zip(pts[:-1], pts[1:]):
            segs.append((d["edge_id"], la, lo, lb, lb2))
    ids = np.array([s[0] for s in segs]); a = np.array([s[1:] for s in segs], float)
    ax, ay = _local_xy(a[:, 0], a[:, 1], lat0, lon0); bx, by = _local_xy(a[:, 2], a[:, 3], lat0, lon0)
    sb = _bearing(ax, ay, bx, by)
    vx, vy = bx - ax, by - ay; vv = np.maximum(vx ** 2 + vy ** 2, 1e-9)
    res_edge, res_status = [], []
    for r in obs.itertuples():
        px, py = _local_xy(r.lat, r.lon, lat0, lon0)
        t = np.clip(((px - ax) * vx + (py - ay) * vy) / vv, 0, 1)
        dist = np.hypot(px - (ax + t * vx), py - (ay + t * vy))
        best = dist.min()
        if best > max_dist_m:
            res_edge.append(np.nan); res_status.append("rejected: farther than max_dist_m"); continue
        cand = np.where(dist <= best + ambiguity_m)[0]
        heading = getattr(r, "heading", np.nan)
        if heading is not None and np.isfinite(heading):
            diff = np.abs((sb[cand] - heading + 180) % 360 - 180)
            cand = cand[diff <= heading_tol_deg]
            if len(cand) == 0:
                res_edge.append(np.nan); res_status.append("rejected: heading matches no nearby edge"); continue
            j = cand[np.argmin(dist[cand])]
            res_edge.append(ids[j]); res_status.append("matched (heading)")
        else:
            uniq = set(ids[cand])
            if len(uniq) == 1:
                res_edge.append(next(iter(uniq))); res_status.append("matched (unique candidate)")
            else:
                res_edge.append(np.nan); res_status.append("rejected: ambiguous direction/road (heading required)")
    out = obs.copy(); out["edge_id"] = res_edge; out["match_status"] = res_status
    return out


def aggregate_matched(matched: pd.DataFrame, freq: str = "15min") -> pd.DataFrame:
    """Median speed per edge and time bin -> input for EdgeSpeedFileProvider."""
    m = matched.dropna(subset=["edge_id"]).copy()
    m["timestamp"] = pd.to_datetime(m["timestamp"]).dt.floor(freq)
    return m.groupby(["edge_id", "timestamp"], as_index=False).agg(speed_kmh=("speed_kmh", "median"), n_obs=("speed_kmh", "size"))
