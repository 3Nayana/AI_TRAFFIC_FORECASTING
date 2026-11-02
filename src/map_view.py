"""Folium map rendering for the route planner (works offline when tiles are disabled)."""
from __future__ import annotations

from typing import Dict, Optional

import folium
import networkx as nx

from . import road_network
from .alternative_routes import PlanResult, ROUTE_COLORS, STRATEGY_LABELS
from .traffic_provider import EdgeSpeed
from .travel_time import LEVEL_COLORS, congestion_index, congestion_level

LEGEND = """
<div style="position: fixed; bottom: 18px; left: 18px; z-index: 9999; background: white; padding: 8px 10px;
 border: 1px solid #888; border-radius: 6px; font-size: 12px; color:#111; line-height:1.5">
<b>Road congestion</b> (speed / free-flow)<br>
<span style="color:#2ca02c">&#9632;</span> free (&gt;80%) &nbsp;<span style="color:#bcbd22">&#9632;</span> light (60-80%)<br>
<span style="color:#ff7f0e">&#9632;</span> moderate (40-60%) &nbsp;<span style="color:#d62728">&#9632;</span> heavy (&lt;40%)<br>
<span style="color:#000">&#9473;&#9473;</span> closed (dashed)<br>
<b>Routes</b>: thick coloured line = candidate; <span style="color:#0057ff"><b>thick blue</b></span> = selected<br>
&#9733; recommended &nbsp; <span style="color:green">&#9679;</span> origin &nbsp; <span style="color:red">&#9679;</span> destination
</div>"""


def build_map(G: nx.DiGraph, speeds: Optional[Dict[str, EdgeSpeed]] = None, plan: Optional[PlanResult] = None,
              selected: int = 0, origin_ll=None, dest_ll=None, tiles: Optional[str] = None, show_network: bool = True) -> folium.Map:
    s, w, n, e = road_network.bounds(G)
    m = folium.Map(location=[(s + n) / 2, (w + e) / 2], zoom_start=15, tiles=tiles, control_scale=True)
    if tiles is None:   # offline: plain background
        m.get_root().html.add_child(folium.Element("<style>.leaflet-container{background:#f4f4f4 !important}</style>"))
    m.fit_bounds([[s, w], [n, e]])
    if show_network:
        drawn = set()
        for u, v, d in G.edges(data=True):
            key = frozenset((u, v)) if (v, u) in G.edges and d["edge_id"] not in drawn else None
            sp = speeds.get(d["edge_id"]) if speeds else None
            if sp is not None and sp.closed:
                col, dash, wt, tip = LEVEL_COLORS["closed"], "6,6", 4, "CLOSED"
            elif sp is not None:
                c = congestion_index(sp.speed_kmh, d["freeflow_kmh"]); lvl = congestion_level(c)
                col, dash, wt, tip = LEVEL_COLORS[lvl], None, 2.5, f"{lvl}: {sp.speed_kmh:.0f} km/h ({sp.kind.value})"
            else:
                col, dash, wt, tip = "#999999", None, 2, "no speed data"
            # offset reverse direction slightly so both directions are visible
            pts = d["geometry"]
            if G.has_edge(v, u) and u > v:
                (la1, lo1), (la2, lo2) = pts[0], pts[-1]
                dx, dy = (lo2 - lo1), (la2 - la1); L = (dx * dx + dy * dy) ** 0.5 or 1
                off = 0.00002
                pts = [(la - dx / L * off, lo + dy / L * off) for la, lo in pts]
            folium.PolyLine(pts, color=col, weight=wt, opacity=0.8, dash_array=dash,
                            tooltip=f"{d.get('name','')} {d['highway']} | limit {d['speed_limit_kmh']:.0f} km/h | {tip}").add_to(m)
    if plan is not None:
        for i, c in enumerate(plan.candidates):
            sel = i == selected
            color = "#0057ff" if sel else ROUTE_COLORS[i % len(ROUTE_COLORS)]
            folium.PolyLine(c.coords, color=color, weight=9 if sel else 5, opacity=0.95 if sel else 0.7,
                            tooltip=f"{c.label}: {c.distance_km:.2f} km, {c.time_min:.1f} min").add_to(m)
        rec = plan.candidates[plan.recommended_index]
        mid = rec.coords[len(rec.coords) // 2]
        folium.Marker(mid, icon=folium.DivIcon(html='<div style="font-size:22px;color:#e6a700;text-shadow:0 0 3px #000">&#9733;</div>'),
                      tooltip="Recommended route").add_to(m)
        o, d = plan.origin, plan.destination
        folium.Marker((G.nodes[o]["y"], G.nodes[o]["x"]), icon=folium.Icon(color="green", icon="play"), tooltip="Origin (snapped)").add_to(m)
        folium.Marker((G.nodes[d]["y"], G.nodes[d]["x"]), icon=folium.Icon(color="red", icon="stop"), tooltip="Destination (snapped)").add_to(m)
    else:
        if origin_ll:
            folium.Marker(origin_ll, icon=folium.Icon(color="green", icon="play"), tooltip="Origin").add_to(m)
        if dest_ll:
            folium.Marker(dest_ll, icon=folium.Icon(color="red", icon="stop"), tooltip="Destination").add_to(m)
    m.get_root().html.add_child(folium.Element(LEGEND))
    return m
