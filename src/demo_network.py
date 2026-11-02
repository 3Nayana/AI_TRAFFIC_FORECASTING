"""Deterministic synthetic demonstration road networks + clearly labelled synthetic traffic scenarios.

The coordinates are ABSTRACT (a local grid placed near lat 0, lon 0); they do not represent any real place.
Scenario speeds are *synthetic*, bounded, and are never labelled observed or predicted.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import networkx as nx
import numpy as np

from .travel_time import MIN_SPEED_KMH

SCENARIOS = {
    "normal": "Normal traffic (mild uniform slowdown)",
    "morning_peak": "Morning peak (inbound towards centre is slow)",
    "evening_peak": "Evening peak (outbound from centre is slow)",
    "heavy_congestion": "Heavy congestion on the central arterials",
    "incident_closure": "Incident: central arterial segments closed + queues nearby",
    "daily_cycle": "Time-dependent demo (speeds vary with hour of day)",
}
MIN_MULT, MAX_MULT = 0.12, 1.0     # documented bounds of the scenario speed multiplier
FREEFLOW_FACTOR = 0.9              # assumed free-flow = 0.9 x speed limit (assumption, not measured)
_M_PER_DEG = 111_320.0


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _add_edge(G: nx.DiGraph, u, v, highway: str, limit: float, oneway: bool, name: str = "") -> None:
    for a, b in ((u, v),) if oneway else ((u, v), (v, u)):
        ya, xa, yb, xb = G.nodes[a]["y"], G.nodes[a]["x"], G.nodes[b]["y"], G.nodes[b]["x"]
        G.add_edge(a, b, edge_id=f"{a}->{b}", length_m=haversine_m(ya, xa, yb, xb), highway=highway,
                   speed_limit_kmh=float(limit), freeflow_kmh=float(limit) * FREEFLOW_FACTOR,
                   oneway=oneway, name=name, geometry=[(ya, xa), (yb, xb)])


def build_grid_network(n: int = 9, spacing_m: float = 350.0) -> nx.DiGraph:
    """n x n grid with a river (few bridges), arterials through the middle, one-way local streets,
    and a fast expressway along the north edge that is slow to enter/leave."""
    G = nx.DiGraph(name="Synthetic grid with river", demo=True)
    d = spacing_m / _M_PER_DEG
    for r in range(n):
        for c in range(n):
            G.add_node(r * n + c, y=(r - n / 2) * d, x=(c - n / 2) * d)
    mid = n // 2
    bridges = {1, mid, n - 2}                       # river between rows mid-1 and mid -> only these columns cross
    for r in range(n):
        for c in range(n):
            u = r * n + c
            if c + 1 < n:                           # horizontal
                hw, lim = ("primary", 60) if r == mid else (("secondary", 50) if r in (1, n - 2) else ("residential", 30))
                if r == n - 1: hw, lim = "motorway", 90
                ow = hw == "residential" and r % 2 == 0 and r not in (0,)
                if ow and (r // 2) % 2 == 1:        # alternate direction
                    _add_edge(G, u + 1, u, hw, lim, True, f"Row {r}")
                else:
                    _add_edge(G, u, u + 1, hw, lim, ow, f"Row {r}")
            if r + 1 < n:                           # vertical
                crossing = r + 1 == mid
                if crossing and c not in bridges:
                    continue
                if r + 1 == n - 1 and c not in (0, n - 1):   # expressway has access only at its two ends
                    continue
                hw, lim = ("primary", 60) if c == mid else (("secondary", 50) if c in (1, n - 2) else ("residential", 30))
                if crossing: hw, lim = "primary", 50
                _add_edge(G, u, u + n, hw, lim, False, f"Col {c}")
    return G


def build_ring_network(rings: int = 4, spokes: int = 12, ring_gap_m: float = 400.0) -> nx.DiGraph:
    """Ring-and-spoke city: centre node, concentric rings, radial spokes (4 are fast arterials)."""
    G = nx.DiGraph(name="Synthetic ring-and-spoke", demo=True)
    G.add_node(0, y=0.0, x=0.0)
    idx = lambda ring, s: 1 + (ring - 1) * spokes + s
    for ring in range(1, rings + 1):
        rad = ring * ring_gap_m / _M_PER_DEG
        for s in range(spokes):
            a = 2 * math.pi * s / spokes
            G.add_node(idx(ring, s), y=rad * math.sin(a), x=rad * math.cos(a))
    for ring in range(1, rings + 1):
        for s in range(spokes):
            hw, lim = ("secondary", 55) if ring == rings else ("residential", 35)
            _add_edge(G, idx(ring, s), idx(ring, (s + 1) % spokes), hw, lim, False, f"Ring {ring}")
            art = s % 3 == 0
            inner = 0 if ring == 1 else idx(ring - 1, s)
            _add_edge(G, inner, idx(ring, s), "primary" if art else "residential", 60 if art else 30, False, f"Spoke {s}")
    return G


DEMO_NETWORKS = {"Synthetic grid with river (9x9)": build_grid_network, "Synthetic ring-and-spoke": build_ring_network}


def _centre(G) -> Tuple[float, float]:
    return (float(np.mean([d["y"] for _, d in G.nodes(data=True)])), float(np.mean([d["x"] for _, d in G.nodes(data=True)])))


def _dist_c(G, n, c) -> float:
    return haversine_m(G.nodes[n]["y"], G.nodes[n]["x"], c[0], c[1])


def scenario_multipliers(G: nx.DiGraph, scenario: str, severity: float = 0.7, hour: float | None = None
                         ) -> Dict[str, Tuple[float, bool]]:
    """Return {edge_id: (speed_multiplier on free-flow, closed)}. Deterministic.

    multiplier is clipped to [MIN_MULT, MAX_MULT]; closure is a separate boolean (edge is removed from routing).
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; choose one of {list(SCENARIOS)}")
    sev = float(min(max(severity, 0.0), 1.0))
    c = _centre(G)
    dmax = max(_dist_c(G, n, c) for n in G.nodes) or 1.0
    out: Dict[str, Tuple[float, bool]] = {}
    arterial_central = []
    for u, v, d in G.edges(data=True):
        mid_d = (_dist_c(G, u, c) + _dist_c(G, v, c)) / 2 / dmax          # 0 centre .. 1 edge
        closeness = 1 - mid_d
        inbound = _dist_c(G, v, c) < _dist_c(G, u, c)
        major = d["highway"] in ("primary", "secondary")
        m, closed = 0.95, False
        if scenario == "normal":
            m = 1.0 - 0.08 * sev * (1 if major else 0.5)
        elif scenario == "morning_peak":
            m = 1.0 - 0.75 * sev * closeness * (1.0 if inbound else 0.15) if d["highway"] != "motorway" else 0.9
        elif scenario == "evening_peak":
            m = 1.0 - 0.75 * sev * closeness * (1.0 if not inbound else 0.15) if d["highway"] != "motorway" else 0.9
        elif scenario == "heavy_congestion":
            m = 1.0 - 0.85 * sev * (closeness ** 0.7) if (d["highway"] == "primary" and closeness > 0.35) else 1.0 - 0.15 * sev * closeness
        elif scenario == "incident_closure":
            if d["highway"] == "primary" and closeness > 0.55:
                arterial_central.append((closeness, u, v))
            m = 1.0 - 0.55 * sev * closeness ** 0.8
        elif scenario == "daily_cycle":
            h = 12.0 if hour is None else float(hour) % 24
            peak = max(math.exp(-((h - 8) ** 2) / 6), math.exp(-((h - 17.5) ** 2) / 6))
            m = 1.0 - 0.8 * sev * peak * closeness * (1.0 if major else 0.5)
        out[d["edge_id"]] = (float(min(max(m, MIN_MULT), MAX_MULT)), closed)
    if scenario == "incident_closure" and arterial_central and sev > 0:
        # close the most central primary segments (both directions), more of them when severity is higher
        n_close = 2 if sev < 0.5 else 4
        pairs = sorted({tuple(sorted((u, v))) for _, u, v in sorted(arterial_central, reverse=True)})[:n_close // 2 or 1]
        for a, b in pairs:
            for x, y in ((a, b), (b, a)):
                if G.has_edge(x, y):
                    out[G[x][y]["edge_id"]] = (MIN_MULT, True)
    return out


def scenario_speed(freeflow_kmh: float, mult: float) -> float:
    return float(max(freeflow_kmh * mult, MIN_SPEED_KMH))
