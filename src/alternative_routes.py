"""Alternative-route engine: strategies, candidate generation (Yen), scoring, time-dependent routing.

OBJECTIVES (all edge-additive, so Dijkstra / Yen are exact for each):
  shortest_distance    J = sum length_m
  fastest_baseline     J = sum t_ff            (assumed free-flow speeds; ignores traffic by design)
  lowest_travel_time   J = sum t               (t from the provider's speeds: scenario / observed / predicted)
  congestion_avoiding  J = sum t * (1 + lambda * c)     c = clip(1 - v/v_ff, 0, 1)
  balanced             J = sum [ w_t * t/T0 + w_d * len/D0 + w_c * (c*t)/T0 ]   (weights normalised to sum 1)
      T0 = fastest route time, D0 = shortest route length under the same speeds (scale references).

Note on double counting: for congestion_avoiding, t already contains the slowdown; the extra lambda*c*t term is a
DELIBERATE risk-aversion penalty (prefer reliably free-flowing roads). lambda=0 reproduces lowest_travel_time.
Road closures are removed from the routing graph (unavailable), never modelled as "very slow".
"""
from __future__ import annotations

import heapq
import logging
import math
import time as _time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import networkx as nx

from . import road_network
from .demo_network import haversine_m
from .traffic_provider import EdgeSpeed, FreeFlowProvider, ProviderStatus, TrafficProvider
from .travel_time import (SpeedKind, congestion_index, congestion_level, sanitize_speed, travel_time_seconds)

log = logging.getLogger(__name__)


class Strategy(str, Enum):
    SHORTEST = "shortest_distance"
    FASTEST_BASELINE = "fastest_baseline"
    LOWEST_TIME = "lowest_travel_time"
    CONGESTION_AVOID = "congestion_avoiding"
    BALANCED = "balanced"


STRATEGY_LABELS = {
    Strategy.SHORTEST: "Shortest distance",
    Strategy.FASTEST_BASELINE: "Fastest baseline (free-flow speeds)",
    Strategy.LOWEST_TIME: "Lowest travel time (traffic-aware)",
    Strategy.CONGESTION_AVOID: "Congestion-avoiding",
    Strategy.BALANCED: "Balanced (time / distance / congestion)",
}
STRATEGY_OBJECTIVE = {
    Strategy.SHORTEST: "minimise total road length",
    Strategy.FASTEST_BASELINE: "minimise travel time at assumed free-flow speeds (ignores traffic)",
    Strategy.LOWEST_TIME: "minimise travel time at the provider's speeds",
    Strategy.CONGESTION_AVOID: "minimise travel time + lambda * congestion-weighted time",
    Strategy.BALANCED: "minimise weighted, normalised time + distance + congestion exposure",
}
ROUTE_COLORS = ["#1f77b4", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#7f7f7f"]


class RoutingError(RuntimeError):
    pass


class InvalidSelectionError(RoutingError):
    pass


class NoRouteError(RoutingError):
    pass


@dataclass
class RouteConfig:
    k: int = 3
    lambda_cong: float = 1.0
    w_time: float = 0.5
    w_dist: float = 0.2
    w_cong: float = 0.3
    congested_c: float = 0.4          # edge is "congested" when speed <= 60 % of free-flow (c >= 0.4)
    overlap_max: float = 0.8          # reject candidate if Jaccard edge-overlap with an accepted route exceeds this
    max_scan: int = 300               # max Yen paths examined while looking for k distinct routes
    max_seconds: float = 15.0
    turn_deg: float = 35.0
    time_dependent: bool = True
    td_bucket_minutes: int = 15

    def norm_weights(self) -> Tuple[float, float, float]:
        w = [max(self.w_time, 0), max(self.w_dist, 0), max(self.w_cong, 0)]
        s = sum(w)
        return (1 / 3, 1 / 3, 1 / 3) if s == 0 else tuple(x / s for x in w)  # type: ignore[return-value]


@dataclass
class RouteResult:
    label: str
    path: List
    edge_ids: List[str]
    coords: List[Tuple[float, float]]
    distance_km: float
    time_min: float
    avg_speed_kmh: float
    n_segments: int
    n_congested: int
    congested_share: float
    exposure_min: float
    turns: int
    objective: Dict[str, float]
    kinds: List[str]
    n_fallback_edges: int
    time_vs_baseline_min: Optional[float] = None
    pct_vs_baseline: Optional[float] = None
    time_vs_shortest_min: Optional[float] = None
    is_recommended: bool = False
    edge_levels: List[str] = field(default_factory=list)
    edge_coords: List[List[Tuple[float, float]]] = field(default_factory=list)


@dataclass
class PlanResult:
    strategy_requested: Strategy
    strategy_used: Strategy
    candidates: List[RouteResult]
    recommended_index: int
    baseline: Optional[RouteResult]
    baseline_note: str
    shortest: Optional[RouteResult]
    status: ProviderStatus
    warnings: List[str]
    fallbacks: List[str]
    explanation: str
    time_saved_assumptions: List[str]
    origin: object = None
    destination: object = None
    network_info: Dict = field(default_factory=dict)


# --------------------------------------------------------------------------- routing graph
def build_routing_graph(G: nx.DiGraph, speeds: Dict[str, EdgeSpeed]) -> Tuple[nx.DiGraph, Dict]:
    """Copy of G with per-edge speed/time/congestion. Closed edges are dropped. Missing/invalid speeds fall back
    to assumed free-flow and are flagged (`fallback=True`)."""
    RG = nx.DiGraph(); RG.add_nodes_from(G.nodes(data=True))
    info = {"closed": 0, "fallback": 0, "edges": G.number_of_edges(), "closed_edges": []}
    for u, v, d in G.edges(data=True):
        sp = speeds.get(d["edge_id"])
        if sp is not None and sp.closed:
            info["closed"] += 1; info["closed_edges"].append(d["edge_id"]); continue
        ff, _ = sanitize_speed(d.get("freeflow_kmh"), fallback=d.get("speed_limit_kmh") or 30.0)
        fb = False
        if sp is None or not (sp.speed_kmh and math.isfinite(sp.speed_kmh) and sp.speed_kmh > 0):
            v_kmh, kind, fb = ff, SpeedKind.FREE_FLOW_ASSUMED, True
        else:
            v_kmh, _ = sanitize_speed(sp.speed_kmh); kind = sp.kind
        info["fallback"] += int(fb)
        t = travel_time_seconds(d["length_m"], v_kmh)
        RG.add_edge(u, v, **{**d, "speed_kmh": v_kmh, "kind": kind.value, "time_s": t,
                             "ff_time_s": travel_time_seconds(d["length_m"], ff), "c": congestion_index(v_kmh, ff), "fallback": fb})
    return RG, info


def edge_cost(strategy: Strategy, e: Dict, cfg: RouteConfig, refs: Dict[str, float]) -> float:
    if strategy == Strategy.SHORTEST:
        return e["length_m"]
    if strategy == Strategy.FASTEST_BASELINE:
        return e["ff_time_s"]
    if strategy == Strategy.LOWEST_TIME:
        return e["time_s"]
    if strategy == Strategy.CONGESTION_AVOID:
        return e["time_s"] * (1.0 + cfg.lambda_cong * e["c"])
    wt, wd, wc = cfg.norm_weights()
    return wt * e["time_s"] / refs["T0"] + wd * e["length_m"] / refs["D0"] + wc * (e["c"] * e["time_s"]) / refs["T0"]


def _weight_fn(strategy, cfg, refs) -> Callable:
    return lambda u, v, d: edge_cost(strategy, d, cfg, refs)


def path_edges(RG: nx.DiGraph, path: List) -> List[Dict]:
    return [RG[a][b] for a, b in zip(path[:-1], path[1:])]


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else 1.0


def generate_candidates(RG: nx.DiGraph, o, d, strategy: Strategy, cfg: RouteConfig, refs: Dict) -> List[List]:
    """Up to cfg.k distinct loopless paths ordered by the strategy objective (Yen's algorithm).
    Near-duplicates (edge Jaccard > overlap_max) are skipped."""
    w = _weight_fn(strategy, cfg, refs)
    out: List[List] = []; sets: List[set] = []; t0 = _time.time()
    try:
        for n, p in enumerate(nx.shortest_simple_paths(RG, o, d, weight=w)):
            es = set(zip(p[:-1], p[1:]))
            if all(_jaccard(es, s) <= cfg.overlap_max for s in sets):
                out.append(p); sets.append(es)
            if len(out) >= cfg.k or n + 1 >= cfg.max_scan or _time.time() - t0 > cfg.max_seconds:
                break
    except nx.NetworkXNoPath as e:
        raise NoRouteError("no directed path from origin to destination") from e
    except nx.NodeNotFound as e:
        raise InvalidSelectionError(str(e)) from e
    return out


# --------------------------------------------------------------------------- time-dependent routing
class SpeedTimeline:
    """Caches provider speeds per time bucket (piece-wise constant in each bucket)."""

    def __init__(self, G, provider: TrafficProvider, bucket_min: int = 15):
        self.G, self.p, self.b, self.cache = G, provider, bucket_min, {}

    def at(self, when: datetime) -> Dict[str, EdgeSpeed]:
        key = when.replace(minute=(when.minute // self.b) * self.b, second=0, microsecond=0)
        if key not in self.cache:
            self.cache[key] = self.p.speeds(self.G, key)
        return self.cache[key]

    def edge(self, d: Dict, when: datetime) -> Optional[Dict]:
        """Edge record valid at `when`; None when closed."""
        sp = self.at(when).get(d["edge_id"])
        ff, _ = sanitize_speed(d.get("freeflow_kmh"), fallback=d.get("speed_limit_kmh") or 30.0)
        if sp is not None and sp.closed:
            return None
        fb = sp is None or not (sp.speed_kmh and math.isfinite(sp.speed_kmh) and sp.speed_kmh > 0)
        v, kind = (ff, SpeedKind.FREE_FLOW_ASSUMED) if fb else (sanitize_speed(sp.speed_kmh)[0], sp.kind)
        return {**d, "speed_kmh": v, "kind": kind.value, "time_s": travel_time_seconds(d["length_m"], v),
                "ff_time_s": travel_time_seconds(d["length_m"], ff), "c": congestion_index(v, ff), "fallback": fb}


def td_dijkstra(G: nx.DiGraph, o, d, depart: datetime, tl: SpeedTimeline) -> Optional[List]:
    """Earliest-arrival path; the speed of an edge is evaluated at the time the vehicle ENTERS it."""
    dist = {o: 0.0}; prev: Dict = {}; pq = [(0.0, 0, o)]; cnt = 1; done = set()
    while pq:
        t, _, u = heapq.heappop(pq)
        if u in done:
            continue
        done.add(u)
        if u == d:
            break
        for v, ed in G[u].items():
            rec = tl.edge(ed, depart + timedelta(seconds=t))
            if rec is None:
                continue
            nt = t + rec["time_s"]
            if nt < dist.get(v, math.inf):
                dist[v] = nt; prev[v] = u; heapq.heappush(pq, (nt, cnt, v)); cnt += 1
    if d not in dist:
        return None
    p = [d]
    while p[-1] != o:
        p.append(prev[p[-1]])
    return p[::-1]


def td_records(G: nx.DiGraph, path: List, depart: datetime, tl: SpeedTimeline) -> Optional[List[Dict]]:
    t = 0.0; recs = []
    for a, b in zip(path[:-1], path[1:]):
        rec = tl.edge(G[a][b], depart + timedelta(seconds=t))
        if rec is None:
            return None
        recs.append(rec); t += rec["time_s"]
    return recs


# --------------------------------------------------------------------------- scoring
def _bearing(p1, p2) -> float:
    la1, lo1, la2, lo2 = map(math.radians, (p1[0], p1[1], p2[0], p2[1]))
    y = math.sin(lo2 - lo1) * math.cos(la2)
    x = math.cos(la1) * math.sin(la2) - math.sin(la1) * math.cos(la2) * math.cos(lo2 - lo1)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def count_turns(recs: List[Dict], threshold_deg: float) -> int:
    n = 0
    for r1, r2 in zip(recs[:-1], recs[1:]):
        g1, g2 = r1.get("geometry"), r2.get("geometry")
        if not g1 or not g2 or len(g1) < 2 or len(g2) < 2:
            continue
        diff = abs((_bearing(g2[0], g2[1]) - _bearing(g1[-2], g1[-1]) + 180) % 360 - 180)
        n += diff >= threshold_deg
    return int(n)


def score_route(recs: List[Dict], path: List, cfg: RouteConfig, refs: Dict, label: str) -> RouteResult:
    dist_m = sum(r["length_m"] for r in recs); t_s = sum(r["time_s"] for r in recs)
    cong = [r for r in recs if r["c"] >= cfg.congested_c]
    coords: List[Tuple[float, float]] = []
    for r in recs:
        g = r["geometry"]; coords.extend(g if not coords else g[1:])
    obj = {s.value: sum(edge_cost(s, r, cfg, refs) for r in recs) for s in Strategy}
    obj["shortest_distance"] /= 1000.0; obj["fastest_baseline"] /= 60.0
    obj["lowest_travel_time"] /= 60.0; obj["congestion_avoiding"] /= 60.0
    return RouteResult(label=label, path=list(path), edge_ids=[r["edge_id"] for r in recs], coords=coords,
                       distance_km=dist_m / 1000, time_min=t_s / 60, avg_speed_kmh=(dist_m / t_s * 3.6) if t_s else 0.0,
                       n_segments=len(recs), n_congested=len(cong), congested_share=len(cong) / len(recs) if recs else 0.0,
                       exposure_min=sum(r["c"] * r["time_s"] for r in recs) / 60, turns=count_turns(recs, cfg.turn_deg),
                       objective=obj, kinds=sorted({r["kind"] for r in recs}), n_fallback_edges=sum(bool(r["fallback"]) for r in recs),
                       edge_levels=[congestion_level(r["c"]) for r in recs], edge_coords=[r["geometry"] for r in recs])


def _refs(RG: nx.DiGraph) -> Dict[str, float]:
    return {"T0": 1.0, "D0": 1.0}


# --------------------------------------------------------------------------- planning
def snap_od(G: nx.DiGraph, origin_latlon, dest_latlon, max_dist_m: float = 1500.0):
    """Snap to nodes inside the largest strongly connected component (so every pair is mutually reachable
    when the graph is one SCC). Raises InvalidSelectionError with a helpful message."""
    try:
        scc = max(nx.strongly_connected_components(G), key=len)
        o, do = road_network.nearest_node(G, *origin_latlon, max_dist_m=max_dist_m, candidates=scc)
        d, dd = road_network.nearest_node(G, *dest_latlon, max_dist_m=max_dist_m, candidates=scc)
    except (ValueError, TypeError, road_network.RoadNetworkError) as e:
        raise InvalidSelectionError(f"Invalid origin/destination: {e}") from e
    if o == d:
        raise InvalidSelectionError("Origin and destination snap to the same road node. Pick points further apart.")
    return o, d, do, dd


def plan_routes(G: nx.DiGraph, origin, destination, provider: Optional[TrafficProvider] = None,
                strategy: Strategy = Strategy.LOWEST_TIME, cfg: Optional[RouteConfig] = None,
                depart: Optional[datetime] = None) -> PlanResult:
    cfg = cfg or RouteConfig(); provider = provider or FreeFlowProvider(); depart = depart or datetime(2026, 1, 12, 8, 0)
    strategy = Strategy(strategy)
    if origin not in G or destination not in G:
        raise InvalidSelectionError("origin/destination node is not in the road graph")
    if origin == destination:
        raise InvalidSelectionError("origin and destination are identical")
    warnings: List[str] = []; fallbacks: List[str] = []
    status = provider.status(G, depart); warnings += status.warnings
    used_provider = provider
    if not status.usable:
        fallbacks.append(f"Traffic source '{status.label}' is not usable ({'; '.join(status.warnings) or 'no data'}). "
                         "Using ASSUMED FREE-FLOW speeds: the result below is NOT traffic-aware.")
        used_provider = FreeFlowProvider()
    is_free = isinstance(used_provider, FreeFlowProvider)
    used = strategy
    if is_free and strategy in (Strategy.LOWEST_TIME, Strategy.CONGESTION_AVOID, Strategy.BALANCED):
        fallbacks.append(f"'{STRATEGY_LABELS[strategy]}' needs traffic speeds; none available -> falling back to "
                         f"'{STRATEGY_LABELS[Strategy.FASTEST_BASELINE]}'. This is not a forecast-based route.")
        used = Strategy.FASTEST_BASELINE
    if strategy == Strategy.LOWEST_TIME and not is_free:
        if status.synthetic:
            warnings.append("'Lowest travel time' is using SYNTHETIC scenario speeds (not a forecast).")
        elif not status.forecast_based:
            warnings.append("'Lowest travel time' is using OBSERVED speeds (current conditions, not a forecast).")

    use_td = cfg.time_dependent and used_provider.is_time_dependent and used in (Strategy.LOWEST_TIME,)
    if used_provider.is_time_dependent and not use_td:
        warnings.append("Provider is time-dependent but this strategy uses a static snapshot at the departure time "
                        "(static approximation: speeds are not updated while driving).")
    speeds0 = used_provider.speeds(G, depart)
    RG, ginfo = build_routing_graph(G, speeds0)
    if ginfo["closed"]:
        warnings.append(f"{ginfo['closed']} directed road segment(s) are CLOSED and unavailable to every route.")
    if ginfo["fallback"]:
        warnings.append(f"{ginfo['fallback']} edge(s) had no valid speed: assumed free-flow used (flagged).")

    # --- reference routes under the current speeds
    try:
        short_p = nx.dijkstra_path(RG, origin, destination, weight="length_m")
        fast_p = nx.dijkstra_path(RG, origin, destination, weight="time_s")
    except nx.NetworkXNoPath as e:
        raise NoRouteError("No route exists between the selected points (disconnected, one-way restrictions or closures). "
                           "Try other points, reduce the incident severity, or load a connected network.") from e
    refs = {"T0": sum(e["time_s"] for e in path_edges(RG, fast_p)) or 1.0,
            "D0": sum(e["length_m"] for e in path_edges(RG, short_p)) or 1.0}

    tl = SpeedTimeline(G, used_provider, cfg.td_bucket_minutes) if use_td else None
    def recs_for(path):
        if tl is not None:
            return td_records(G, path, depart, tl)
        return path_edges(RG, path)

    cand_paths = generate_candidates(RG, origin, destination, used, cfg, refs)
    if use_td:
        tdp = td_dijkstra(G, origin, destination, depart, tl)
        if tdp and tdp not in cand_paths:
            cand_paths.insert(0, tdp)
    cands: List[RouteResult] = []
    for p in cand_paths:
        r = recs_for(p)
        if r:
            cands.append(score_route(r, p, cfg, refs, ""))
    if not cands:
        raise NoRouteError("Candidate generation produced no valid route.")
    key = used.value
    order = sorted(range(len(cands)), key=lambda i: cands[i].objective[key])   # selection by the declared objective
    cands = [cands[i] for i in order]
    for i, c in enumerate(cands):
        c.label = f"Route {i + 1}" + (" (recommended)" if i == 0 else f" (alternative {i})")
    cands[0].is_recommended = True

    # --- baseline: fastest route at ASSUMED FREE-FLOW speeds on the unrestricted graph (what a no-traffic navigator picks)
    ffG = nx.DiGraph(); ffG.add_nodes_from(G.nodes(data=True))
    for u, v, d in G.edges(data=True):
        ff, _ = sanitize_speed(d.get("freeflow_kmh"), fallback=30.0)
        ffG.add_edge(u, v, **d, ff_t=travel_time_seconds(d["length_m"], ff))
    base_p = nx.dijkstra_path(ffG, origin, destination, weight="ff_t")
    base_recs = recs_for(base_p) if tl is not None else (path_edges(RG, base_p) if all(RG.has_edge(a, b) for a, b in zip(base_p[:-1], base_p[1:])) else None)
    baseline, bnote = None, ""
    if base_recs is None:
        bnote = ("The no-traffic baseline route passes through a CLOSED road, so it is infeasible under this scenario; "
                 "time saved versus the baseline cannot be computed (the baseline would be impassable).")
    else:
        baseline = score_route(base_recs, base_p, cfg, refs, "Baseline (free-flow fastest)")
        for c in cands:
            c.time_vs_baseline_min = c.time_min - baseline.time_min
            c.pct_vs_baseline = 100 * (c.time_min - baseline.time_min) / baseline.time_min if baseline.time_min else None
    shortest = None
    sr = recs_for(short_p)
    if sr:
        shortest = score_route(sr, short_p, cfg, refs, "Shortest distance")
        for c in cands:
            c.time_vs_shortest_min = c.time_min - shortest.time_min

    rec = cands[0]
    expl = (f"Selected by the declared objective for '{STRATEGY_LABELS[used]}' ({STRATEGY_OBJECTIVE[used]}): "
            f"it has the lowest objective value among {len(cands)} distinct candidate(s) "
            f"({rec.objective[key]:.2f} vs next best "
            f"{(cands[1].objective[key] if len(cands) > 1 else float('nan')):.2f}). "
            f"{rec.distance_km:.2f} km, {rec.time_min:.1f} min, {rec.n_congested}/{rec.n_segments} congested segments.")
    assumptions = [
        "Times come from edge length / provider speed; they are model estimates, not measured journey times.",
        f"Speed source for this plan: {status.label if status.usable else 'ASSUMED FREE-FLOW (fallback)'}.",
        "Baseline = fastest route at assumed free-flow speeds (0.9 x speed limit), evaluated under the SAME current speeds.",
        "Time saved = baseline time - recommended time under identical speeds; positive means the recommendation is faster.",
        "No intersection/turn delays, signals, or driver behaviour are modelled." + (" Speeds sampled when entering each edge." if use_td else ""),
    ]
    return PlanResult(strategy, used, cands, 0, baseline, bnote, shortest, status, warnings, fallbacks, expl, assumptions,
                      origin, destination, ginfo)
