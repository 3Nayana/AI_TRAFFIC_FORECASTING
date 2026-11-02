"""Small deterministic fixtures. TEST-ONLY data; never used by the app."""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from src.demo_network import haversine_m, FREEFLOW_FACTOR


def diamond(extra_oneway: bool = False) -> nx.DiGraph:
    """A -> B -> D (short, 2 km)  and  A -> C -> D (longer, 2.4 km); two-way except where noted."""
    G = nx.DiGraph(name="diamond", demo=True)
    coords = {"A": (0.0, 0.0), "B": (0.009, 0.004), "C": (-0.011, 0.004), "D": (0.0, 0.009)}   # (lat, lon)
    for n, (la, lo) in coords.items():
        G.add_node(n, y=la, x=lo)
    def add(u, v, limit=50.0, oneway=False):
        pairs = [(u, v)] if oneway else [(u, v), (v, u)]
        for a, b in pairs:
            ya, xa, yb, xb = G.nodes[a]["y"], G.nodes[a]["x"], G.nodes[b]["y"], G.nodes[b]["x"]
            G.add_edge(a, b, edge_id=f"{a}->{b}", length_m=haversine_m(ya, xa, yb, xb), highway="primary", speed_limit_kmh=limit,
                       freeflow_kmh=limit * FREEFLOW_FACTOR, oneway=oneway, name="", geometry=[(ya, xa), (yb, xb)])
    add("A", "B"); add("B", "D"); add("A", "C"); add("C", "D", oneway=extra_oneway)
    return G


def synthetic_traffic_csv(days: int = 60, seed: int = 0) -> pd.DataFrame:
    """Fake hourly counts for 4 junctions (TEST ONLY) in the same schema as the Kaggle file."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2017-01-01", periods=days * 24, freq="h")
    rows = []
    for j in (1, 2, 3, 4):
        base = 20 * j + 15 * np.sin(2 * np.pi * (idx.hour - 6) / 24) + 8 * (idx.dayofweek < 5)
        v = np.clip(base + rng.normal(0, 3, len(idx)), 1, None).round()
        rows.append(pd.DataFrame({"DateTime": idx.strftime("%Y-%m-%d %H:%M:%S"), "Junction": j, "Vehicles": v,
                                  "ID": [int(t.strftime("%Y%m%d%H")) * 10 + j for t in idx]}))
    return pd.concat(rows, ignore_index=True)
