from datetime import datetime, timedelta

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from src import road_network as rn
from src import traffic_provider as tp
from src.travel_time import SpeedKind
from tests.helpers import diamond

T0 = datetime(2026, 1, 12, 8, 0)


# ------------------------------------------------------------ geographic edge-to-traffic matching
def _obs(lat, lon, speed=30.0, heading=np.nan):
    return pd.DataFrame({"lat": [lat], "lon": [lon], "speed_kmh": [speed], "timestamp": [T0], "heading": [heading]})


def test_match_oneway_edge_without_heading():
    G = diamond(extra_oneway=True)            # C->D is one-way
    g = G["C"]["D"]["geometry"]; mid = ((g[0][0] + g[1][0]) / 2, (g[0][1] + g[1][1]) / 2)
    m = tp.match_observations_to_edges(G, _obs(*mid))
    assert m.edge_id.iloc[0] == "C->D"


def test_two_way_street_needs_heading():
    G = diamond()
    g = G["A"]["B"]["geometry"]; mid = ((g[0][0] + g[1][0]) / 2, (g[0][1] + g[1][1]) / 2)
    amb = tp.match_observations_to_edges(G, _obs(*mid))
    assert np.isnan(amb.edge_id.iloc[0]) and "ambiguous" in amb.match_status.iloc[0]
    fwd = tp.match_observations_to_edges(G, _obs(*mid, heading=tp._bearing(0, 0, *(np.array(tp._local_xy(g[1][0], g[1][1], g[0][0], g[0][1])))))) \
        if False else None
    from src.alternative_routes import _bearing
    hAB = _bearing(g[0], g[1]); hBA = (hAB + 180) % 360
    assert tp.match_observations_to_edges(G, _obs(*mid, heading=hAB)).edge_id.iloc[0] == "A->B"
    assert tp.match_observations_to_edges(G, _obs(*mid, heading=hBA)).edge_id.iloc[0] == "B->A"


def test_far_points_are_rejected():
    G = diamond()
    m = tp.match_observations_to_edges(G, _obs(0.5, 0.5))
    assert np.isnan(m.edge_id.iloc[0]) and "farther" in m.match_status.iloc[0]


def test_aggregate_and_file_provider_roundtrip():
    G = diamond(extra_oneway=True)
    g = G["C"]["D"]["geometry"]; mid = ((g[0][0] + g[1][0]) / 2, (g[0][1] + g[1][1]) / 2)
    obs = pd.concat([_obs(*mid, s) for s in (20.0, 30.0, 40.0)], ignore_index=True)
    agg = tp.aggregate_matched(tp.match_observations_to_edges(G, obs))
    assert agg.speed_kmh.iloc[0] == 30.0
    p = tp.EdgeSpeedFileProvider(agg, SpeedKind.OBSERVED, max_age_minutes=15, min_coverage=0.0)
    sp = p.speeds(G, T0 + timedelta(minutes=5))
    assert sp["C->D"].speed_kmh == 30.0 and sp["C->D"].kind == SpeedKind.OBSERVED


# ------------------------------------------------------------ provider validation
def test_file_provider_validation_and_units():
    G = diamond()
    with pytest.raises(ValueError):
        tp.EdgeSpeedFileProvider(pd.DataFrame({"edge_id": ["A->B"]}))
    with pytest.raises(ValueError):
        tp.EdgeSpeedFileProvider(pd.DataFrame({"edge_id": [], "timestamp": [], "speed_kmh": []}), SpeedKind.SPEED_LIMIT)
    df = pd.DataFrame({"edge_id": ["A->B", "B->D", "ghost"], "timestamp": [T0] * 3, "speed_kmh": [10.0, -3.0, 20.0]})
    p = tp.EdgeSpeedFileProvider(df, SpeedKind.OBSERVED, speed_unit="mph", min_coverage=0.0)
    sp = p.speeds(G, T0)
    assert sp["A->B"].speed_kmh == pytest.approx(16.09344) and "B->D" not in sp     # negative speed dropped
    st = p.status(G, T0)
    assert any("not geographically matched" in w for w in st.warnings) and p.n_bad_rows == 1


def test_low_coverage_makes_provider_unusable():
    G = diamond()
    df = pd.DataFrame({"edge_id": ["A->B"], "timestamp": [T0], "speed_kmh": [20.0]})
    st = tp.EdgeSpeedFileProvider(df, SpeedKind.OBSERVED, min_coverage=0.5).status(G, T0)
    assert not st.usable and st.coverage == pytest.approx(1 / G.number_of_edges())


def test_predicted_uses_nearest_forecast_within_tolerance():
    G = diamond()
    df = pd.DataFrame({"edge_id": ["A->B"] * 2, "timestamp": [T0, T0 + timedelta(hours=1)], "speed_kmh": [20.0, 40.0]})
    p = tp.EdgeSpeedFileProvider(df, SpeedKind.PREDICTED, forecast_tolerance_minutes=20, min_coverage=0.0)
    assert p.speeds(G, T0 + timedelta(minutes=50))["A->B"].speed_kmh == 40.0
    assert p.speeds(G, T0 + timedelta(minutes=30)) == {}                           # outside tolerance -> not used
    assert p.status(G, T0).forecast_based


def test_live_provider_never_claims_live(monkeypatch):
    monkeypatch.delenv("TRAFFIC_API_KEY", raising=False)
    st = tp.LiveAPIProvider().status(diamond(), T0)
    assert not st.usable and any("NOT available" in w for w in st.warnings)


# ------------------------------------------------------------ road network standardisation
def _osm_like():
    M = nx.MultiDiGraph(name="x", crs="EPSG:4326")
    for n, (y, x) in {1: (0, 0), 2: (0, 0.002), 3: (0.002, 0.002)}.items():
        M.add_node(n, y=y, x=x)
    M.add_edge(1, 2, 0, length=222.0, highway="residential", maxspeed="30 mph", name="Main")
    M.add_edge(1, 2, 1, length=300.0, highway="residential")                    # longer parallel edge -> dropped
    M.add_edge(2, 1, 0, length=222.0, highway="residential")
    M.add_edge(2, 3, 0, highway=["primary", "secondary"], maxspeed=["50", "40"])  # one-way, no length, list tags
    return M


def test_standardize_preserves_direction_and_attributes():
    G = rn.standardize_graph(_osm_like())
    assert isinstance(G, nx.DiGraph) and G.has_edge(2, 3) and not G.has_edge(3, 2)      # one-way kept
    assert G[1][2]["length_m"] == 222.0                                                # shortest parallel kept
    assert G[1][2]["speed_limit_kmh"] == pytest.approx(48.28, abs=0.01)                # 30 mph
    assert G[2][3]["speed_limit_kmh"] == 40.0 and G[2][3]["length_m"] > 100             # lowest list value, haversine length
    assert G[2][3]["oneway"] is True and len(G[1][2]["geometry"]) == 2


def test_standardize_rejects_missing_coordinates():
    M = nx.MultiDiGraph(); M.add_node(1); M.add_node(2, x=0, y=0); M.add_edge(1, 2)
    with pytest.raises(rn.RoadNetworkError):
        rn.standardize_graph(M)


def test_graphml_roundtrip_and_cache_missing(tmp_path):
    M = _osm_like()
    for _, _, d in M.edges(data=True):
        d["maxspeed"] = str(d.get("maxspeed", "")) if not isinstance(d.get("maxspeed"), list) else "50"
        d["highway"] = d["highway"] if isinstance(d["highway"], str) else "primary"
    nx.write_graphml(M, tmp_path / "g.graphml")
    G = rn.load_graphml(tmp_path / "g.graphml")
    assert G.number_of_nodes() == 3 and G.has_edge(2, 3)
    with pytest.raises(rn.RoadNetworkError):
        rn.load_graphml(tmp_path / "missing.graphml")
    with pytest.raises(rn.RoadNetworkError):
        rn.download_graph()


def test_parse_maxspeed():
    assert rn.parse_maxspeed("50") == 50 and rn.parse_maxspeed("signals") is None and rn.parse_maxspeed(None) is None
    assert rn.parse_maxspeed(["50", "30"]) == 30
