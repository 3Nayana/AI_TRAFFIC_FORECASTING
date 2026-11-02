from datetime import datetime, timedelta

import networkx as nx
import pytest

from src import alternative_routes as ar
from src import demo_network as dn
from src.alternative_routes import (InvalidSelectionError, NoRouteError, RouteConfig, Strategy, plan_routes, snap_od,
                                    edge_cost)
from src.traffic_provider import (EdgeSpeed, EdgeSpeedFileProvider, FreeFlowProvider, ProviderStatus, ScenarioProvider,
                                  TrafficProvider)
from src.travel_time import SpeedKind
from tests.helpers import diamond

T0 = datetime(2026, 1, 12, 8, 0)


class DictProvider(TrafficProvider):
    """Test provider: explicit speeds per edge (default = free-flow), optional closures / time rule."""
    name = "dict"

    def __init__(self, slow=None, closed=(), td_rule=None, kind=SpeedKind.SCENARIO_SYNTHETIC):
        self.slow, self.closed, self.td_rule, self.kind = slow or {}, set(closed), td_rule, kind
        self.is_time_dependent = td_rule is not None

    def speeds(self, G, when):
        out = {}
        for _, _, d in G.edges(data=True):
            e = d["edge_id"]; v = self.slow.get(e, d["freeflow_kmh"])
            if self.td_rule and self.td_rule(e, when):
                v = 5.0
            out[e] = EdgeSpeed(v, self.kind, e in self.closed, when)
        return out

    def status(self, G, when):
        return ProviderStatus(self.kind, "TEST", True, synthetic=True, time_dependent=self.is_time_dependent)


# ---------------------------------------------------------------- the key scenario from the specification
def test_congestion_makes_shortest_route_slower_and_router_picks_alternative():
    G = diamond()
    free = plan_routes(G, "A", "D", DictProvider(), Strategy.LOWEST_TIME, RouteConfig(k=2), T0)
    assert free.candidates[0].path == ["A", "B", "D"]                         # normal shortest = fastest
    jam = DictProvider(slow={"B->D": 8.0})
    r = plan_routes(G, "A", "D", jam, Strategy.LOWEST_TIME, RouteConfig(k=2), T0)
    shortest = plan_routes(G, "A", "D", jam, Strategy.SHORTEST, RouteConfig(k=2), T0)
    assert shortest.candidates[0].path == ["A", "B", "D"]                     # distance ignores the jam ...
    assert r.shortest.time_min > r.candidates[0].time_min                     # ... and is now slower
    assert r.candidates[0].path == ["A", "C", "D"]                            # traffic-aware avoids it
    assert r.baseline.path == ["A", "B", "D"] and r.candidates[0].time_vs_baseline_min < 0
    assert r.candidates[0].pct_vs_baseline < 0


@pytest.mark.parametrize("strategy", list(Strategy))
@pytest.mark.parametrize("slow", [{}, {"B->D": 8.0}, {"A->C": 12.0, "B->D": 30.0}])
def test_recommended_route_is_optimal_for_declared_objective(strategy, slow):
    G = diamond(); cfg = RouteConfig(k=5, lambda_cong=2.0)
    res = plan_routes(G, "A", "D", DictProvider(slow=slow), strategy, cfg, T0)
    RG, _ = ar.build_routing_graph(G, DictProvider(slow=slow).speeds(G, T0))
    refs = {"T0": min(sum(e["time_s"] for e in ar.path_edges(RG, p)) for p in nx.all_simple_paths(RG, "A", "D")),
            "D0": min(sum(e["length_m"] for e in ar.path_edges(RG, p)) for p in nx.all_simple_paths(RG, "A", "D"))}
    used = res.strategy_used
    costs = {tuple(p): sum(edge_cost(used, e, cfg, refs) for e in ar.path_edges(RG, p)) for p in nx.all_simple_paths(RG, "A", "D")}
    best = min(costs.values())
    rec = res.candidates[res.recommended_index]
    assert costs[tuple(rec.path)] == pytest.approx(best)
    assert all(costs[tuple(res.candidates[0].path)] <= costs[tuple(c.path)] + 1e-9 for c in res.candidates)


def test_congestion_penalty_changes_choice_only_when_lambda_large():
    G = diamond()
    prov = DictProvider(slow={"B->D": 40.0})        # mildly congested, still slightly faster in pure time? compute both
    t = plan_routes(G, "A", "D", prov, Strategy.LOWEST_TIME, RouteConfig(k=2), T0)
    c0 = plan_routes(G, "A", "D", prov, Strategy.CONGESTION_AVOID, RouteConfig(k=2, lambda_cong=0.0), T0)
    c9 = plan_routes(G, "A", "D", prov, Strategy.CONGESTION_AVOID, RouteConfig(k=2, lambda_cong=50.0), T0)
    assert c0.candidates[0].path == t.candidates[0].path == ["A", "B", "D"]      # lambda=0 == lowest travel time
    assert c9.candidates[0].path == ["A", "C", "D"]                              # strong penalty avoids the congested edge
    assert c9.candidates[0].exposure_min < c0.candidates[0].exposure_min


# ---------------------------------------------------------------- closures, one-way, connectivity
def test_closed_edge_is_unavailable_not_slow():
    G = diamond()
    prov = DictProvider(closed={"B->D"})
    RG, info = ar.build_routing_graph(G, prov.speeds(G, T0))
    assert not RG.has_edge("B", "D") and info["closed"] == 1
    r = plan_routes(G, "A", "D", prov, Strategy.SHORTEST, RouteConfig(k=3), T0)
    assert all("B->D" not in c.edge_ids for c in r.candidates) and r.candidates[0].path == ["A", "C", "D"]
    assert r.baseline is None and "CLOSED" in r.baseline_note        # no-traffic baseline would be impassable


def test_all_routes_closed_raises_noroute_with_advice():
    G = diamond()
    with pytest.raises(NoRouteError, match="No route exists"):
        plan_routes(G, "A", "D", DictProvider(closed={"B->D", "C->D"}), Strategy.LOWEST_TIME, RouteConfig(), T0)


def test_one_way_is_enforced():
    G = diamond(extra_oneway=True)                  # C->D exists, D->C does not
    assert G.has_edge("C", "D") and not G.has_edge("D", "C")
    r = plan_routes(G, "D", "A", DictProvider(), Strategy.SHORTEST, RouteConfig(k=5), T0)
    for c in r.candidates:
        assert all(G.has_edge(a, b) for a, b in zip(c.path[:-1], c.path[1:]))
        assert "D->C" not in c.edge_ids
    assert [c.path for c in r.candidates] == [["D", "B", "A"]]    # the only legal way back
    G.remove_edge("B", "D"); G.remove_edge("D", "B")
    with pytest.raises(NoRouteError):
        plan_routes(G, "D", "A", DictProvider(), Strategy.SHORTEST, RouteConfig(), T0)


def test_disconnected_destination_and_invalid_selections():
    G = diamond(); G.add_node("E", y=0.05, x=0.05)
    with pytest.raises(NoRouteError):
        plan_routes(G, "A", "E", DictProvider(), Strategy.LOWEST_TIME, RouteConfig(), T0)
    with pytest.raises(InvalidSelectionError):
        plan_routes(G, "A", "A", DictProvider(), Strategy.LOWEST_TIME, RouteConfig(), T0)
    with pytest.raises(InvalidSelectionError):
        plan_routes(G, "A", "nope", DictProvider(), Strategy.LOWEST_TIME, RouteConfig(), T0)


def test_snapping_errors():
    G = diamond()
    with pytest.raises(InvalidSelectionError, match="outside network coverage"):
        snap_od(G, (45.0, 45.0), (0.0, 0.0))
    with pytest.raises(InvalidSelectionError):
        snap_od(G, (float("nan"), 0.0), (0.0, 0.009))
    with pytest.raises(InvalidSelectionError, match="same road node"):
        snap_od(G, (0.0, 0.0), (0.0001, 0.0001))
    o, d, _, _ = snap_od(G, (0.0001, 0.0), (0.0, 0.0089))
    assert (o, d) == ("A", "D")


# ---------------------------------------------------------------- alternatives and de-duplication
def test_alternatives_are_distinct_valid_and_limited_by_topology():
    G = diamond()
    r = plan_routes(G, "A", "D", DictProvider(), Strategy.LOWEST_TIME, RouteConfig(k=5), T0)
    paths = [tuple(c.path) for c in r.candidates]
    assert len(paths) == len(set(paths)) == 2                    # only two loopless paths exist
    assert all(len(set(p)) == len(p) for p in paths)             # loopless
    assert r.candidates[0].time_min <= r.candidates[1].time_min


def test_near_duplicate_routes_are_removed():
    G = diamond()                                                # add a tiny detour sharing most edges with A-B-D
    G.add_node("X", y=0.0091, x=0.0045)
    for a, b in (("B", "X"), ("X", "D")):
        G.add_edge(a, b, edge_id=f"{a}->{b}", length_m=700.0, highway="primary", speed_limit_kmh=50.0, freeflow_kmh=45.0,
                   oneway=True, name="", geometry=[(G.nodes[a]["y"], G.nodes[a]["x"]), (G.nodes[b]["y"], G.nodes[b]["x"])])
    strict = plan_routes(G, "A", "D", DictProvider(), Strategy.LOWEST_TIME, RouteConfig(k=5, overlap_max=0.2), T0)
    loose = plan_routes(G, "A", "D", DictProvider(), Strategy.LOWEST_TIME, RouteConfig(k=5, overlap_max=0.99), T0)
    assert len(loose.candidates) == 3 and len(strict.candidates) == 2
    for a in strict.candidates:
        for b in strict.candidates:
            if a is not b:
                ea, eb = set(a.edge_ids), set(b.edge_ids)
                assert len(ea & eb) / len(ea | eb) <= 0.2


# ---------------------------------------------------------------- honest fallbacks
def test_unavailable_strategy_falls_back_visibly():
    G = diamond()
    r = plan_routes(G, "A", "D", FreeFlowProvider(), Strategy.LOWEST_TIME, RouteConfig(), T0)
    assert r.strategy_requested == Strategy.LOWEST_TIME and r.strategy_used == Strategy.FASTEST_BASELINE
    assert any("not a forecast-based route" in f for f in r.fallbacks)


def test_stale_observations_are_not_used_and_say_so():
    import pandas as pd
    G = diamond()
    df = pd.DataFrame({"edge_id": ["B->D"], "timestamp": [T0 - timedelta(hours=3)], "speed_kmh": [5.0]})
    p = EdgeSpeedFileProvider(df, SpeedKind.OBSERVED, max_age_minutes=15)
    r = plan_routes(G, "A", "D", p, Strategy.LOWEST_TIME, RouteConfig(), T0)
    assert not r.status.usable and r.fallbacks and r.candidates[0].path == ["A", "B", "D"]   # stale jam ignored, disclosed


def test_synthetic_scenarios_warn_and_are_labelled():
    G = dn.build_grid_network()
    r = plan_routes(G, 0, 80, ScenarioProvider("heavy_congestion", 0.8), Strategy.LOWEST_TIME, RouteConfig(), T0)
    assert r.status.synthetic and any("SYNTHETIC" in w for w in r.warnings)
    assert set(k for c in r.candidates for k in c.kinds) == {"scenario_synthetic"}


# ---------------------------------------------------------------- time-dependent routing
def test_time_dependent_uses_speed_at_edge_entry_time():
    G = diamond()
    slow_after = lambda e, when: e == "B->D" and when >= T0 + timedelta(seconds=30)    # jam starts shortly after departure
    prov = DictProvider(td_rule=slow_after)
    cfg_td = RouteConfig(k=2, td_bucket_minutes=1, time_dependent=True)
    cfg_static = RouteConfig(k=2, td_bucket_minutes=1, time_dependent=False)
    td = plan_routes(G, "A", "D", prov, Strategy.LOWEST_TIME, cfg_td, T0)
    st = plan_routes(G, "A", "D", prov, Strategy.LOWEST_TIME, cfg_static, T0)
    assert st.candidates[0].path == ["A", "B", "D"]            # snapshot at departure sees no jam (static approximation)
    assert any("static approximation" in w for w in st.warnings)
    assert td.candidates[0].path == ["A", "C", "D"]            # vehicle reaches B->D after the jam started


# ---------------------------------------------------------------- demo scenarios
@pytest.mark.parametrize("scn", list(dn.SCENARIOS))
@pytest.mark.parametrize("sev", [0.0, 0.5, 1.0])
def test_scenario_speeds_are_bounded(scn, sev):
    G = dn.build_grid_network()
    m = dn.scenario_multipliers(G, scn, sev, hour=8)
    assert set(m) == {d["edge_id"] for _, _, d in G.edges(data=True)}
    assert all(dn.MIN_MULT <= v <= dn.MAX_MULT for v, _ in m.values())
    sp = ScenarioProvider(scn, sev).speeds(G, T0)
    assert all(s.speed_kmh >= 5.0 for s in sp.values())


def test_incident_closure_closes_edges_and_no_route_uses_them():
    G = dn.build_grid_network(); prov = ScenarioProvider("incident_closure", 0.9)
    sp = prov.speeds(G, T0); closed = {k for k, s in sp.items() if s.closed}
    assert closed
    for o, d in [(0, 80), (8, 72), (36, 44), (4, 76)]:
        try:
            r = plan_routes(G, o, d, prov, Strategy.LOWEST_TIME, RouteConfig(k=3), T0)
        except NoRouteError:
            continue
        assert all(not (set(c.edge_ids) & closed) for c in r.candidates)


def test_scenarios_change_the_recommended_route_on_the_demo_graph():
    G = dn.build_grid_network()
    changed = 0
    for o, d in [(0, 80), (8, 72), (3, 77), (9, 71), (27, 53), (1, 79), (5, 75), (18, 62)]:
        n = plan_routes(G, o, d, ScenarioProvider("normal", 0.9), Strategy.LOWEST_TIME, RouteConfig(), T0).candidates[0].path
        h = plan_routes(G, o, d, ScenarioProvider("heavy_congestion", 0.9), Strategy.LOWEST_TIME, RouteConfig(), T0).candidates[0].path
        changed += n != h
    assert changed >= 2


def test_route_metrics_consistency():
    G = diamond()
    r = plan_routes(G, "A", "D", DictProvider(slow={"B->D": 10.0}), Strategy.LOWEST_TIME, RouteConfig(k=2), T0)
    for c in r.candidates:
        assert c.distance_km == pytest.approx(sum(G[a][b]["length_m"] for a, b in zip(c.path[:-1], c.path[1:])) / 1000)
        assert c.avg_speed_kmh == pytest.approx(c.distance_km / (c.time_min / 60))
        assert 0 <= c.congested_share <= 1 and c.n_segments == len(c.path) - 1
    jam = [c for c in r.candidates if "B->D" in c.edge_ids][0]
    assert jam.n_congested == 1 and jam.exposure_min > 0


def test_turn_counting():
    recs = [dict(geometry=[(0, 0), (0, 0.001)]), dict(geometry=[(0, 0.001), (0.001, 0.001)]), dict(geometry=[(0.001, 0.001), (0.002, 0.001)])]
    assert ar.count_turns(recs, 35) == 1          # east -> north is a turn; north -> north is straight
