"""Routing experiments. ALL results here are SIMULATED on synthetic scenarios (ground truth = the scenario speeds).
They show how the algorithm behaves; they are NOT real-world validation. Real validation needs observed journey times.
"""
from __future__ import annotations

import logging
import random
from datetime import datetime
from typing import Dict, Iterable, List, Optional

import networkx as nx
import numpy as np
import pandas as pd

from . import config
from .alternative_routes import (RouteConfig, Strategy, NoRouteError, build_routing_graph, path_edges, plan_routes)
from .demo_network import SCENARIOS
from .traffic_provider import EdgeSpeed, ScenarioProvider, TrafficProvider, ProviderStatus
from .travel_time import SpeedKind

log = logging.getLogger(__name__)
DEPART = datetime(2026, 1, 12, 8, 0)
TOL = 0.005   # a route must be >0.5 % faster to count as "faster" (avoids float ties)


def _jacc(a, b) -> float:
    A, B = set(zip(a[:-1], a[1:])), set(zip(b[:-1], b[1:]))
    return len(A & B) / len(A | B) if A | B else 1.0


def sample_pairs(G: nx.DiGraph, n: int, seed: int = 0, min_hops: int = 4) -> List[tuple]:
    rng = random.Random(seed); nodes = sorted(G.nodes); out = []
    while len(out) < n:
        o, d = rng.sample(nodes, 2)
        if nx.shortest_path_length(G, o, d) >= min_hops:
            out.append((o, d))
    return out


def run_scenario_experiment(G: nx.DiGraph, scenarios: Iterable[str] = tuple(SCENARIOS), severities=(0.3, 0.6, 0.9),
                            n_pairs: int = 60, seed: int = 0) -> pd.DataFrame:
    pairs = sample_pairs(G, n_pairs, seed); rows = []
    for sc in scenarios:
        for sev in severities:
            prov = ScenarioProvider(sc, sev)
            for o, d in pairs:
                try:
                    r = plan_routes(G, o, d, prov, Strategy.LOWEST_TIME, RouteConfig(k=3), DEPART)
                except NoRouteError:
                    continue
                rec, base, sh = r.candidates[0], r.baseline, r.shortest
                rows.append(dict(scenario=sc, severity=sev, origin=o, destination=d, t_traffic_aware=rec.time_min,
                                 t_static_fastest=base.time_min if base else np.nan,
                                 t_shortest=sh.time_min if sh else np.nan, baseline_blocked=base is None,
                                 same_route_as_static=bool(base and base.path == rec.path),
                                 route_overlap=_jacc(rec.path, base.path) if base else np.nan,
                                 pct_saved_vs_static=100 * (base.time_min - rec.time_min) / base.time_min if base else np.nan,
                                 pct_saved_vs_shortest=100 * (sh.time_min - rec.time_min) / sh.time_min if sh else np.nan))
    return pd.DataFrame(rows)


def summarize_scenarios(df: pd.DataFrame) -> pd.DataFrame:
    def f(g):
        feas = g[~g.baseline_blocked]
        return pd.Series({
            "trips": len(g),
            "baseline_blocked_%": 100 * g.baseline_blocked.mean(),
            "route_differs_from_static_%": 100 * (1 - feas.same_route_as_static.mean()) if len(feas) else np.nan,
            "faster_than_static_%": 100 * (feas.t_traffic_aware < feas.t_static_fastest * (1 - TOL)).mean() if len(feas) else np.nan,
            "faster_than_shortest_%": 100 * (g.t_traffic_aware < g.t_shortest * (1 - TOL)).mean(),
            "mean_saved_vs_static_%": feas.pct_saved_vs_static.mean() if len(feas) else np.nan,
            "mean_saved_vs_shortest_%": g.pct_saved_vs_shortest.mean(),
            "mean_time_traffic_aware_min": g.t_traffic_aware.mean(),
        })
    return df.groupby(["scenario", "severity"]).apply(f, include_groups=False).reset_index()


class _Noisy(TrafficProvider):
    """The planner sees speeds x lognormal noise (a stand-in for forecast uncertainty); closures stay exact."""
    def __init__(self, base: ScenarioProvider, sigma: float, seed: int):
        self.base, self.sigma, self.seed = base, sigma, seed; self.name = "noisy"

    def speeds(self, G, when):
        rng = np.random.default_rng(self.seed); out = {}
        for k, s in sorted(self.base.speeds(G, when).items()):
            n = float(np.exp(rng.normal(0, self.sigma))) if self.sigma > 0 else 1.0
            out[k] = EdgeSpeed(max(s.speed_kmh * n, 5.0), s.kind, s.closed, s.timestamp)
        return out

    def status(self, G, when): return self.base.status(G, when)


def run_noise_experiment(G, scenario="heavy_congestion", severity=0.8, sigmas=(0.0, 0.1, 0.2, 0.4, 0.8),
                         n_pairs: int = 40, reps: int = 3, seed: int = 0) -> pd.DataFrame:
    """Plan with perturbed speeds, evaluate under the true scenario speeds. Regret = extra time vs the oracle route."""
    truth = ScenarioProvider(scenario, severity); RG, _ = build_routing_graph(G, truth.speeds(G, DEPART))
    pairs = sample_pairs(G, n_pairs, seed); rows = []
    for sg in sigmas:
        for rep in range(reps if sg > 0 else 1):
            for i, (o, d) in enumerate(pairs):
                try:
                    oracle = nx.dijkstra_path(RG, o, d, weight="time_s")
                    plan = plan_routes(G, o, d, _Noisy(truth, sg, seed * 1000 + rep * 100 + i), Strategy.LOWEST_TIME, RouteConfig(k=1), DEPART)
                except (NoRouteError, nx.NetworkXNoPath):
                    continue
                p = plan.candidates[0].path
                if not all(RG.has_edge(a, b) for a, b in zip(p[:-1], p[1:])):
                    continue
                t = lambda path: sum(e["time_s"] for e in path_edges(RG, path)) / 60
                base = plan.baseline
                bt = t(base.path) if base and all(RG.has_edge(a, b) for a, b in zip(base.path[:-1], base.path[1:])) else np.nan
                rows.append(dict(sigma=sg, rep=rep, t_chosen=t(p), t_oracle=t(oracle), t_static=bt, same_as_oracle=p == oracle))
    df = pd.DataFrame(rows)
    df["regret_%"] = 100 * (df.t_chosen - df.t_oracle) / df.t_oracle
    df["saved_vs_static_%"] = 100 * (df.t_static - df.t_chosen) / df.t_static
    return df.groupby("sigma").agg(trips=("t_chosen", "size"), mean_regret_pct=("regret_%", "mean"),
                                   pct_same_as_oracle=("same_as_oracle", lambda x: 100 * x.mean()),
                                   mean_saved_vs_static_pct=("saved_vs_static_%", "mean")).reset_index()


def scenario_route_table(G: nx.DiGraph, o, d, severity: float = 0.8, strategies=tuple(Strategy)) -> pd.DataFrame:
    rows = []
    for sc in SCENARIOS:
        for st in strategies:
            try:
                r = plan_routes(G, o, d, ScenarioProvider(sc, severity), st, RouteConfig(k=3), DEPART)
            except NoRouteError:
                continue
            c = r.candidates[0]
            rows.append(dict(scenario=sc, strategy=st.value, distance_km=c.distance_km, time_min=c.time_min,
                             exposure_min=c.exposure_min, turns=c.turns, n_congested=c.n_congested,
                             path_hash=hash(tuple(c.path)) % 10**6, path=",".join(map(str, c.path))))
    return pd.DataFrame(rows)


def evaluate_predicted_vs_observed(df: pd.DataFrame) -> Dict[str, float]:
    """Real evaluation helper. df columns: predicted_min, observed_min (one row per completed trip)."""
    need = {"predicted_min", "observed_min"}
    if not need <= set(df.columns):
        raise ValueError(f"need columns {sorted(need)}")
    d = df.dropna(subset=list(need)); d = d[d.observed_min > 0]
    if d.empty:
        raise ValueError("no valid rows")
    err = d.predicted_min - d.observed_min
    return {"n": int(len(d)), "MAE_min": float(err.abs().mean()), "RMSE_min": float(np.sqrt((err ** 2).mean())),
            "MAPE_%": float(100 * (err.abs() / d.observed_min).mean()), "bias_min": float(err.mean())}


def run_all(n_pairs: int = 60) -> None:
    from .demo_network import build_grid_network, build_ring_network
    out = config.REPORTS_DIR / "routing"; out.mkdir(parents=True, exist_ok=True)
    for name, G in (("grid", build_grid_network()), ("ring", build_ring_network())):
        df = run_scenario_experiment(G, n_pairs=n_pairs); df.to_csv(out / f"scenario_trips_{name}.csv", index=False)
        summarize_scenarios(df).to_csv(out / f"scenario_summary_{name}.csv", index=False)
        run_noise_experiment(G, n_pairs=max(20, n_pairs // 2)).to_csv(out / f"noise_summary_{name}.csv", index=False)
        log.info("routing experiments done for %s", name)


if __name__ == "__main__":
    config.setup_logging(); run_all()
