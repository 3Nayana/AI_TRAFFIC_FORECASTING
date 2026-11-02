"""Road-network loading, standardisation, caching and point snapping.

All routing code uses a *standard* nx.DiGraph with node attrs x(lon), y(lat) and edge attrs:
edge_id, length_m, highway, speed_limit_kmh, freeflow_kmh, oneway, name, geometry[(lat, lon), ...].
Direction is preserved (a two-way street = two directed edges; a one-way street = one).
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Iterable, Optional, Tuple

import networkx as nx

from . import config
from .demo_network import DEMO_NETWORKS, FREEFLOW_FACTOR, haversine_m

log = logging.getLogger(__name__)

DEFAULT_LIMITS = {"motorway": 100, "trunk": 80, "primary": 60, "secondary": 50, "tertiary": 40,
                  "unclassified": 35, "residential": 30, "living_street": 20, "service": 20}


class RoadNetworkError(RuntimeError):
    pass


def parse_maxspeed(value) -> Optional[float]:
    """OSM maxspeed ('50', '30 mph', ['50','30'], 'signals') -> km/h or None."""
    if isinstance(value, (list, tuple)):
        vals = [parse_maxspeed(v) for v in value]
        vals = [v for v in vals if v]
        return min(vals) if vals else None
    if value is None:
        return None
    s = str(value).strip().lower()
    try:
        if "mph" in s:
            return float(s.replace("mph", "").strip()) * 1.609344
        return float(s.split()[0])
    except (ValueError, IndexError):
        return None


def _first(v):
    return v[0] if isinstance(v, (list, tuple)) and v else v


def standardize_graph(M: nx.Graph) -> nx.DiGraph:
    """Convert an OSMnx-style MultiDiGraph (or any graph) into the standard DiGraph.

    Parallel edges u->v are collapsed to the *shortest* one (documented simplification).
    Edges without 'length' get a haversine length from node coordinates; missing speeds use class defaults.
    """
    if not M.is_directed():
        M = M.to_directed()
    G = nx.DiGraph(name=M.graph.get("name", "road network"), demo=False, crs=M.graph.get("crs", "EPSG:4326"))
    for n, d in M.nodes(data=True):
        if "x" not in d or "y" not in d:
            raise RoadNetworkError(f"node {n} has no coordinates (x/y); cannot route geographically")
        G.add_node(n, x=float(d["x"]), y=float(d["y"]))
    best = {}
    edges = M.edges(keys=True, data=True) if M.is_multigraph() else ((u, v, 0, d) for u, v, d in M.edges(data=True))
    for u, v, _k, d in edges:
        if u == v:
            continue
        ya, xa, yb, xb = G.nodes[u]["y"], G.nodes[u]["x"], G.nodes[v]["y"], G.nodes[v]["x"]
        length = d.get("length")
        try:
            length = float(length)
        except (TypeError, ValueError):
            length = haversine_m(ya, xa, yb, xb)
        if not math.isfinite(length) or length <= 0:
            length = max(haversine_m(ya, xa, yb, xb), 1.0)
        hw = str(_first(d.get("highway", "unclassified")))
        limit = parse_maxspeed(d.get("maxspeed")) or DEFAULT_LIMITS.get(hw, 30)
        geom = d.get("geometry")
        if geom is not None and hasattr(geom, "coords"):
            pts = [(lat, lon) for lon, lat in geom.coords]
        elif isinstance(geom, list):
            pts = geom
        else:
            pts = [(ya, xa), (yb, xb)]
        rec = dict(edge_id=f"{u}->{v}", length_m=length, highway=hw, speed_limit_kmh=float(limit),
                   freeflow_kmh=float(limit) * FREEFLOW_FACTOR, oneway=bool(d.get("oneway", not M.has_edge(v, u))),
                   name=str(_first(d.get("name", "")) or ""), geometry=pts)
        if (u, v) not in best or length < best[(u, v)]["length_m"]:
            best[(u, v)] = rec
    for (u, v), rec in best.items():
        G.add_edge(u, v, **rec)
    if G.number_of_edges() == 0:
        raise RoadNetworkError("graph has no usable edges")
    return G


def largest_scc_subgraph(G: nx.DiGraph) -> nx.DiGraph:
    """Keep the largest strongly connected component so every node can reach every other node."""
    comp = max(nx.strongly_connected_components(G), key=len)
    return G.subgraph(comp).copy()


# ------------------------------------------------------------------ loading
def get_demo_graph(name: str) -> nx.DiGraph:
    if name not in DEMO_NETWORKS:
        raise RoadNetworkError(f"unknown demo network {name!r}")
    return DEMO_NETWORKS[name]()


def load_graphml(path: str | Path) -> nx.DiGraph:
    """Load a user-provided GraphML (e.g. saved by osmnx.save_graphml)."""
    path = Path(path)
    if not path.exists():
        raise RoadNetworkError(f"{path} not found")
    try:
        import osmnx as ox
        M = ox.load_graphml(path)
    except ImportError:
        log.warning("osmnx not installed: using networkx.read_graphml (string attributes are converted best-effort)")
        M = nx.read_graphml(path)
        if all(str(n).lstrip("-").isdigit() for n in M.nodes):      # GraphML stores ids as text; OSM ids are ints
            M = nx.relabel_nodes(M, {n: int(n) for n in M.nodes})
        for _, d in M.nodes(data=True):
            for k in ("x", "y"):
                if k in d:
                    d[k] = float(d[k])
    return standardize_graph(M)


def download_graph(place: Optional[str] = None, point: Optional[Tuple[float, float]] = None, dist_m: int = 1500,
                   network_type: str = "drive", cache_dir: Path = config.ROAD_CACHE_DIR) -> nx.DiGraph:
    """Download a road network for a place name OR (lat, lon, dist). Needs internet + osmnx. Cached as GraphML."""
    if not place and not point:
        raise RoadNetworkError("provide either a place name or a (lat, lon) point")
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(json.dumps([place, point, dist_m, network_type]).encode()).hexdigest()[:16]
    f = cache_dir / f"{key}.graphml"
    if f.exists():
        log.info("using cached road graph %s", f)
        return load_graphml(f)
    try:
        import osmnx as ox
    except ImportError as e:
        raise RoadNetworkError("osmnx is not installed (pip install osmnx)") from e
    try:
        M = ox.graph_from_place(place, network_type=network_type) if place else \
            ox.graph_from_point(point, dist=dist_m, network_type=network_type)
    except Exception as e:  # network / geocoding errors
        raise RoadNetworkError(f"download failed ({type(e).__name__}: {e}). Check internet access or load a GraphML file.") from e
    ox.save_graphml(M, f)
    return load_graphml(f)


# ------------------------------------------------------------------ snapping
def nearest_node(G: nx.DiGraph, lat: float, lon: float, max_dist_m: float = 1500.0,
                 candidates: Optional[Iterable] = None) -> Tuple[object, float]:
    if not (isinstance(lat, (int, float)) and isinstance(lon, (int, float)) and math.isfinite(lat) and math.isfinite(lon)):
        raise ValueError("coordinates must be finite numbers")
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError(f"coordinates out of range: ({lat}, {lon})")
    nodes = list(candidates) if candidates is not None else list(G.nodes)
    if not nodes:
        raise RoadNetworkError("graph has no nodes")
    best = min(nodes, key=lambda n: haversine_m(lat, lon, G.nodes[n]["y"], G.nodes[n]["x"]))
    dist = haversine_m(lat, lon, G.nodes[best]["y"], G.nodes[best]["x"])
    if dist > max_dist_m:
        raise ValueError(f"point is {dist:.0f} m from the nearest road node (limit {max_dist_m:.0f} m): outside network coverage")
    return best, dist


def bounds(G: nx.DiGraph) -> Tuple[float, float, float, float]:
    ys = [d["y"] for _, d in G.nodes(data=True)]; xs = [d["x"] for _, d in G.nodes(data=True)]
    return min(ys), min(xs), max(ys), max(xs)
