"""AI Traffic Forecasting and Smart Route Recommendation System - Streamlit app.   Run:  streamlit run app.py"""
from __future__ import annotations

import json
import os
from datetime import date, datetime, time as dtime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import config, congestion, data_loader, road_network, routing_evaluation
from src.alternative_routes import (InvalidSelectionError, NoRouteError, RouteConfig, STRATEGY_LABELS, STRATEGY_OBJECTIVE,
                                    Strategy, plan_routes, snap_od)
from src.demo_network import DEMO_NETWORKS, SCENARIOS
from src.map_view import build_map
from src.traffic_provider import EdgeSpeedFileProvider, FreeFlowProvider, LiveAPIProvider, ScenarioProvider
from src.travel_time import SpeedKind

st.set_page_config(page_title="AI Traffic Forecasting & Route Recommendation", page_icon="🚦", layout="wide")
PAGES = ["1 Overview", "2 Traffic analysis", "3 Traffic forecasting", "4 Route planner", "5 Route comparison & experiments",
         "6 Research evaluation", "7 Data sources & settings"]


# ------------------------------------------------------------------ cached loaders
@st.cache_data(show_spinner=False)
def load_json(path: str):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data(show_spinner=False)
def load_csv(path: str, **kw):
    p = Path(path)
    return pd.read_csv(p, **kw) if p.exists() else None


@st.cache_resource(show_spinner=False)
def demo_graph(name: str):
    return road_network.get_demo_graph(name)


def missing_artifact(what: str):
    st.info(f"{what} not found. Run the pipeline first:  `python run_pipeline.py`  (needs data/raw/traffic.csv).")


def mode_banner():
    ss = st.session_state
    st.caption(f"Operating mode: traffic input = **{ss.get('traffic_mode', 'Synthetic scenario (demo)')}** | road network = "
               f"**{ss.get('net_label', 'demo')}**. Synthetic scenario speeds are illustrative, not real traffic, and are NOT outputs of the Kaggle-trained model.")


# ------------------------------------------------------------------ pages
def page_overview():
    st.title("🚦 AI Traffic Forecasting & Smart Route Recommendation")
    mode_banner()
    st.warning("The Kaggle dataset has vehicle COUNTS for four anonymous junctions with no coordinates or road links. "
               "Forecasts therefore cannot drive routing directly. The route planner runs on a demonstration network with clearly labelled "
               "SYNTHETIC traffic scenarios until geographically matched edge speeds are supplied (page 7).")
    a = load_json(str(config.PROCESSED_DIR / "audit.json"))
    if not a:
        return missing_artifact("Audit report")
    c = st.columns(4)
    c[0].metric("Rows", f"{a['rows']:,}"); c[1].metric("Junctions", len(a["junctions"]))
    c[2].metric("Start", a["start"][:10]); c[3].metric("End", a["end"][:10])
    st.subheader("Junction coverage")
    rows = [{"Junction": j, "Rows": v["rows"], "From": v["start"][:16], "To": v["end"][:16], "Interval": v["modal_interval"],
             "Missing hours": v["missing_timestamps"], "Gaps": v["n_gaps"], "Mean veh/h": v["vehicles"]["mean"], "Max": v["vehicles"]["max"]}
            for j, v in a["junctions"].items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(f"Missing values per column: {a['missing_values']} | duplicate rows: {a['duplicate_rows']}")
    m = load_csv(str(config.REPORTS_DIR / "forecast_metrics.csv"))
    st.subheader("Model performance (test period, mean over junctions)")
    if m is None:
        missing_artifact("Forecast metrics")
    else:
        t = m.groupby(["model", "horizon"]).MAE.mean().unstack().round(2)
        main = [x for x in t.index if x in ("Persistence", "SeasonalNaive-24h", "SeasonalNaive-168h", "HistoricalMean(dow,hour)", "Ridge",
                                            "RandomForest", "LSTM_ensemble")]
        st.dataframe(t.loc[main].rename(columns=lambda h: f"MAE +{h}h"), width="stretch")
    with st.expander("Dataset limitations"):
        for l in a.get("limitations", []):
            st.write("• " + l)


def page_analysis():
    st.title("Traffic analysis")
    df = load_csv(str(config.PROCESSED_DIR / "traffic_clean.csv"), parse_dates=["DateTime"])
    a = load_json(str(config.PROCESSED_DIR / "audit.json"))
    if df is None:
        return missing_artifact("Processed traffic data")
    js = sorted(df.Junction.unique()); sel = st.multiselect("Junctions", js, default=js)
    res = st.radio("Aggregation", ["Hourly", "Daily mean", "Weekly mean"], horizontal=True, index=1)
    d = df[df.Junction.isin(sel)]
    rule = {"Hourly": None, "Daily mean": "D", "Weekly mean": "W"}[res]
    if rule:
        d = d.set_index("DateTime").groupby("Junction").Vehicles.resample(rule).mean().reset_index()
    st.plotly_chart(px.line(d, x="DateTime", y="Vehicles", color=d.Junction.astype(str), title="Vehicle counts"), width="stretch")
    c1, c2 = st.columns(2)
    dd = df[df.Junction.isin(sel)].assign(hour=lambda x: x.DateTime.dt.hour, weekday=lambda x: x.DateTime.dt.day_name())
    hp = dd.groupby(["hour", "Junction"]).Vehicles.mean().reset_index(); hp["Junction"] = hp["Junction"].astype(str)
    c1.plotly_chart(px.line(hp, x="hour", y="Vehicles", color="Junction", title="Mean by hour of day"), width="stretch")
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    w = dd.groupby(["weekday", "Junction"]).Vehicles.mean().reset_index()
    c2.plotly_chart(px.bar(w, x="weekday", y="Vehicles", color=w.Junction.astype(str), barmode="group",
                           category_orders={"weekday": order}, title="Mean by weekday"), width="stretch")
    st.plotly_chart(px.box(dd, x="Junction", y="Vehicles", title="Distribution of hourly counts"), width="stretch")
    st.subheader("Data quality checks")
    if a:
        q = pd.DataFrame([{"Junction": j, "Rows": v["rows"], "Expected if complete": v["expected_rows_if_complete"],
                           "Missing timestamps": v["missing_timestamps"], "Gaps": v["n_gaps"], "Interval counts": str(v["interval_counts"])}
                          for j, v in a["junctions"].items()])
        st.dataframe(q, hide_index=True, width="stretch")
        st.caption(f"Duplicates: {a['duplicate_rows']} rows, {a['duplicate_junction_timestamp']} duplicate (junction, timestamp) pairs. Nulls: {a['missing_values']}.")


def page_forecast():
    st.title("Traffic forecasting")
    pred = load_csv(str(config.PROCESSED_DIR / "predictions_test.csv.gz"), parse_dates=["origin_time", "target_time"])
    m = load_csv(str(config.REPORTS_DIR / "forecast_metrics.csv")); cfg = load_json(str(config.REPORTS_DIR / "training_config.json"))
    if pred is None or m is None:
        return missing_artifact("Predictions / metrics")
    st.caption("Predicted VEHICLE COUNTS per junction. Relative congestion classes below describe how busy a junction is versus its own "
               "training history - not speed or delay. Counts are never converted to travel times.")
    c = st.columns(3)
    j = c[0].selectbox("Junction", sorted(pred.junction.unique())); h = c[1].selectbox("Horizon (hours ahead)", sorted(pred.horizon.unique()))
    models = sorted(pred.model.unique()); default = [x for x in ("SeasonalNaive-168h", "RandomForest", "LSTM_ensemble") if x in models]
    chosen = c[2].multiselect("Models", models, default=default)
    g = pred[(pred.junction == j) & (pred.horizon == h)]
    lo, hi = g.target_time.min().to_pydatetime(), g.target_time.max().to_pydatetime()
    rng = st.slider("Test-period window", lo, hi, (lo, min(lo + pd.Timedelta(days=7), hi)))
    g = g[(g.target_time >= rng[0]) & (g.target_time <= rng[1])]
    fig = go.Figure(); a0 = g[g.model == (chosen[0] if chosen else models[0])].sort_values("target_time")
    fig.add_scatter(x=a0.target_time, y=a0.actual, name="actual", line=dict(color="black", width=2))
    for mm in chosen:
        gm = g[g.model == mm].sort_values("target_time"); fig.add_scatter(x=gm.target_time, y=gm.pred, name=mm, mode="lines")
    fig.update_layout(title=f"Junction {j}, +{h}h: actual vs predicted vehicles/hour", yaxis_title="vehicles/hour")
    st.plotly_chart(fig, width="stretch")
    st.subheader("Per-junction metrics")
    mm = m[(m.junction == j) & (m.horizon == h)].sort_values("MAE")[["model", "n", "MAE", "RMSE", "R2"] + [c for c in m.columns if c.startswith("MAE_skill")]]
    st.dataframe(mm.round(3), hide_index=True, width="stretch")
    st.plotly_chart(px.bar(m[m.horizon == h].groupby("model").MAE.mean().reset_index().sort_values("MAE"), x="MAE", y="model",
                           orientation="h", title=f"Mean MAE across junctions, +{h}h (lower is better)"), width="stretch")
    st.subheader("Predicted counts and relative congestion")
    if cfg and chosen:
        thr = {int(k): v for k, v in cfg["congestion_thresholds"].items()}
        sel_model = st.selectbox("Model for congestion table", chosen)
        t = g[g.model == sel_model].sort_values("target_time").copy()
        t["relative_congestion"] = congestion.classify(t.pred, j, t.target_time.dt.hour, thr)
        t["actual_class"] = congestion.classify(t.actual, j, t.target_time.dt.hour, thr)
        t[["pred", "actual"]] = t[["pred", "actual"]].round(1)
        st.dataframe(t[["target_time", "pred", "relative_congestion", "actual", "actual_class"]].rename(columns={"pred": "predicted_vehicles", "actual": "actual_vehicles"}),
                     hide_index=True, width="stretch", height=260)
        th = thr[int(j)]["all"]
        st.caption(f"Thresholds for junction {j} (train-period quantiles {cfg['congestion_quantiles']}): {[round(x, 1) for x in th]} vehicles/h. {congestion.LIMITATIONS}")
        acc = (t.relative_congestion == t.actual_class).mean()
        st.metric("Class agreement (predicted vs actual class)", f"{acc:.1%}")


def _graph_ui():
    ss = st.session_state
    src = st.selectbox("Road network source", list(DEMO_NETWORKS) + ["Upload GraphML", "Download via OSMnx (needs internet)"], key="net_src")
    G = None
    if src in DEMO_NETWORKS:
        G = demo_graph(src); ss["net_label"] = f"{src} (synthetic demo)"
    elif src == "Upload GraphML":
        up = st.file_uploader("GraphML file (e.g. saved with osmnx.save_graphml)", type=["graphml", "xml"])
        if up:
            tmp = config.ROAD_CACHE_DIR; tmp.mkdir(parents=True, exist_ok=True)
            f = tmp / f"upload_{abs(hash(up.name))}.graphml"; f.write_bytes(up.getvalue())
            try:
                G = road_network.largest_scc_subgraph(road_network.load_graphml(f)); ss["net_label"] = f"{up.name} (user graph)"
            except Exception as e:
                st.error(f"Could not load graph: {e}")
    else:
        place = st.text_input("Place name (any location)", key="osm_place"); pt = st.text_input("...or 'lat, lon' + radius below", key="osm_pt")
        dist = st.number_input("Radius (m) for lat/lon", 300, 5000, 1500)
        if st.button("Download / load from cache"):
            try:
                point = tuple(float(x) for x in pt.split(",")) if pt.strip() else None
                with st.spinner("Downloading road network..."):
                    ss["osm_graph"] = road_network.largest_scc_subgraph(road_network.download_graph(place or None, point, int(dist)))
                ss["net_label"] = f"OSM: {place or pt}"
            except Exception as e:
                st.error(f"{e}")
        G = ss.get("osm_graph")
    return G


def _provider_ui(G, depart):
    ss = st.session_state
    modes = ["Synthetic scenario (demo)", "Edge-speed file (observed / predicted)", "Assumed free-flow only", "Live traffic API"]
    mode = st.selectbox("Traffic input mode", modes, key="traffic_mode")
    if mode.startswith("Synthetic"):
        if not G.graph.get("demo"):
            st.warning("Synthetic scenarios are designed for the demonstration networks; on a real graph they are applied by distance from the centroid.")
        sc = st.selectbox("Scenario", list(SCENARIOS), format_func=lambda k: SCENARIOS[k], key="scenario")
        sev = st.slider("Congestion severity", 0.0, 1.0, 0.8, 0.05, key="severity")
        return ScenarioProvider(sc, sev)
    if mode.startswith("Edge-speed"):
        up = st.file_uploader("CSV: edge_id, timestamp, speed_kmh[, closed]", type="csv")
        kind = st.radio("These speeds are", ["observed", "predicted"], horizontal=True)
        unit = st.selectbox("Speed unit", ["kmh", "mph", "ms"]); age = st.number_input("Max age of observations (min)", 1, 240, 15)
        tpl = pd.DataFrame({"edge_id": [d["edge_id"] for _, _, d in G.edges(data=True)], "timestamp": depart.isoformat(),
                            "speed_kmh": [round(d["freeflow_kmh"], 1) for _, _, d in G.edges(data=True)]})
        st.download_button("Template CSV (free-flow placeholders - replace with real data)", tpl.to_csv(index=False), "edge_speeds_template.csv")
        if up is None:
            st.info("Upload edge-level speeds that are already matched to this graph's edge_ids. Until then no traffic data is used.")
            return FreeFlowProvider()
        try:
            return EdgeSpeedFileProvider(pd.read_csv(up), SpeedKind(kind), max_age_minutes=age, speed_unit=unit)
        except Exception as e:
            st.error(f"Invalid file: {e}"); return FreeFlowProvider()
    if mode.startswith("Live"):
        return LiveAPIProvider()
    return FreeFlowProvider()


def page_planner():
    ss = st.session_state
    st.title("Route planner")
    # apply a click recorded in the previous run BEFORE widgets are created
    pend = ss.pop("_pending_click", None)
    if pend:
        k = "o" if pend["mode"] == "Set origin" else "d"
        ss[f"{k}_lat"], ss[f"{k}_lon"] = round(pend["lat"], 6), round(pend["lng"], 6)
    left, right = st.columns([1, 2])
    with left:
        with st.expander("1 · Road network", expanded=False):
            G = _graph_ui()
        if G is None:
            G = demo_graph(list(DEMO_NETWORKS)[0]); ss.setdefault("net_label", list(DEMO_NETWORKS)[0] + " (synthetic demo)")
        s, w, n, e = road_network.bounds(G)
        ss.setdefault("o_lat", round(s + 0.15 * (n - s), 6)); ss.setdefault("o_lon", round(w + 0.15 * (e - w), 6))
        ss.setdefault("d_lat", round(s + 0.85 * (n - s), 6)); ss.setdefault("d_lon", round(w + 0.85 * (e - w), 6))
        st.markdown("**2 · Origin / destination**")
        ss["click_mode"] = st.radio("Map click sets", ["Set origin", "Set destination", "Off"], horizontal=True, index=ss.get("click_idx", 0))
        c = st.columns(2)
        c[0].number_input("Origin lat", format="%.6f", key="o_lat"); c[1].number_input("Origin lon", format="%.6f", key="o_lon")
        c[0].number_input("Dest lat", format="%.6f", key="d_lat"); c[1].number_input("Dest lon", format="%.6f", key="d_lon")
        if os.environ.get("ENABLE_GEOCODING") == "1":
            q = st.text_input("Address search (Nominatim, needs internet)")
            if q and st.button("Find address"):
                try:
                    from geopy.geocoders import Nominatim
                    loc = Nominatim(user_agent=os.environ.get("GEOCODER_USER_AGENT", "traffic-demo")).geocode(q, timeout=8)
                    st.write(loc.address if loc else "No match") if not loc else ss.update(d_lat=loc.latitude, d_lon=loc.longitude)
                except Exception as ex:
                    st.error(f"Geocoding failed: {ex}")
        else:
            st.caption("Address search is disabled (set ENABLE_GEOCODING=1 and install geopy; needs internet). Use map clicks or coordinates.")
        st.markdown("**3 · Traffic & departure**")
        d0 = st.date_input("Departure date", date.today(), key="dep_date"); t0 = st.time_input("Departure time", dtime(8, 0), key="dep_time")
        depart = datetime.combine(d0, t0)
        provider = _provider_ui(G, depart)
        st.markdown("**4 · Strategy**")
        strat = st.selectbox("Routing strategy", list(Strategy), format_func=lambda s_: STRATEGY_LABELS[s_], index=2, key="strategy")
        st.caption("Objective: " + STRATEGY_OBJECTIVE[strat])
        k = st.slider("Number of alternatives", 1, 5, 3)
        lam, wt, wd, wc, ov = 1.0, 0.5, 0.2, 0.3, 0.8
        with st.expander("Advanced objective settings"):
            lam = st.slider("Congestion penalty lambda (congestion-avoiding)", 0.0, 5.0, 1.0, 0.1)
            wt = st.slider("Balanced: time weight", 0.0, 1.0, 0.5, 0.05); wd = st.slider("Balanced: distance weight", 0.0, 1.0, 0.2, 0.05)
            wc = st.slider("Balanced: congestion weight", 0.0, 1.0, 0.3, 0.05); ov = st.slider("Max edge overlap between alternatives", 0.3, 1.0, 0.8, 0.05)
            td = st.checkbox("Use time-dependent routing when the provider supports it", True)
        go_btn = st.button("Calculate alternative routes", type="primary", width="stretch")
        if go_btn:
            try:
                o, d, do, dd = snap_od(G, (ss["o_lat"], ss["o_lon"]), (ss["d_lat"], ss["d_lon"]))
                ss["plan"] = plan_routes(G, o, d, provider, strat, RouteConfig(k=k, lambda_cong=lam, w_time=wt, w_dist=wd, w_cong=wc,
                                                                               overlap_max=ov, time_dependent=td), depart)
                ss["plan"].network_info["snap_m"] = (round(do), round(dd)); ss["plan_provider_speeds"] = provider.speeds(G, depart)
                ss["sel_route"] = 0; ss["plan_error"] = None
            except (InvalidSelectionError, NoRouteError) as ex:
                ss["plan"], ss["plan_error"] = None, str(ex)
            except Exception as ex:                       # never crash the page
                ss["plan"], ss["plan_error"] = None, f"Unexpected error: {type(ex).__name__}: {ex}"
    plan = ss.get("plan")
    with right:
        tiles = "OpenStreetMap" if st.checkbox("Show OpenStreetMap tiles (needs internet; abstract demo coordinates have no real map)", False) else None
        speeds = ss.get("plan_provider_speeds") if plan else provider.speeds(G, depart)
        m = build_map(G, speeds, plan, ss.get("sel_route", 0), (ss["o_lat"], ss["o_lon"]), (ss["d_lat"], ss["d_lon"]), tiles)
        from streamlit_folium import st_folium
        out = st_folium(m, height=560, use_container_width=True, key="planner_map", returned_objects=["last_clicked"])
        lc = (out or {}).get("last_clicked")
        if lc and ss["click_mode"] != "Off" and lc != ss.get("_last_click"):
            ss["_last_click"] = lc; ss["_pending_click"] = {"mode": ss["click_mode"], "lat": lc["lat"], "lng": lc["lng"]}
            ss["click_idx"] = ["Set origin", "Set destination", "Off"].index(ss["click_mode"]); st.rerun()
    if ss.get("plan_error"):
        st.error(ss["plan_error"])
        st.info("Recovery: pick points closer to the network, make sure both lie on connected roads, lower the incident severity, "
                "or choose another scenario/network. One-way streets and closures can make a destination unreachable.")
    if plan:
        show_plan(plan, G)


def show_plan(plan, G):
    ss = st.session_state
    for f in plan.fallbacks:
        st.error("FALLBACK: " + f)
    for w in plan.warnings:
        st.warning(w)
    st.success(f"Recommended: Route 1 for **{STRATEGY_LABELS[plan.strategy_used]}**. " + plan.explanation)
    rec = plan.candidates[0]
    c = st.columns(4)
    c[0].metric("Distance", f"{rec.distance_km:.2f} km"); c[1].metric("Estimated time", f"{rec.time_min:.1f} min")
    if rec.time_vs_baseline_min is not None:
        c[2].metric("Time vs free-flow baseline", f"{-rec.time_vs_baseline_min:+.1f} min saved", f"{-rec.pct_vs_baseline:+.1f}%")
    else:
        c[2].metric("Time vs free-flow baseline", "n/a")
    c[3].metric("Congestion exposure", f"{rec.exposure_min:.2f} min")
    if plan.baseline is None:
        st.info(plan.baseline_note)
    with st.expander("How is 'time saved' estimated? (assumptions)"):
        for a in plan.time_saved_assumptions:
            st.write("• " + a)
    key = plan.strategy_used.value
    rows = []
    for i, r in enumerate(plan.candidates):
        rows.append({"Route": r.label, "Distance (km)": round(r.distance_km, 2), "Time (min)": round(r.time_min, 1),
                     "Avg speed (km/h)": round(r.avg_speed_kmh, 1), "Congested segs": f"{r.n_congested}/{r.n_segments} ({r.congested_share:.0%})",
                     "Exposure (min)": round(r.exposure_min, 2), "Turns": r.turns,
                     "vs baseline (min)": None if r.time_vs_baseline_min is None else round(r.time_vs_baseline_min, 2),
                     "vs baseline (%)": None if r.pct_vs_baseline is None else round(r.pct_vs_baseline, 1),
                     f"Objective ({key})": round(r.objective[key], 3), "Speed source": ", ".join(r.kinds)})
    df = pd.DataFrame(rows); st.dataframe(df, hide_index=True, width="stretch")
    st.download_button("Download comparison CSV", df.to_csv(index=False), "route_comparison.csv")
    labels = [r.label for r in plan.candidates]
    pick = st.radio("Highlight a candidate on the map", labels, horizontal=True, index=min(ss.get("sel_route", 0), len(labels) - 1))
    new = labels.index(pick)
    if new != ss.get("sel_route", 0):
        ss["sel_route"] = new; st.rerun()
    st.caption("Route colours: selected route is thick blue; others use distinct colours. Strategy legend: " +
               " | ".join(f"{STRATEGY_LABELS[s]}: {STRATEGY_OBJECTIVE[s]}" for s in Strategy))


def page_experiments():
    st.title("Route comparison & experiments")
    st.error("SIMULATED results on synthetic scenarios (ground truth = the scenario's own speeds). They show how routing behaves; they are NOT real-world validation.")
    net = st.selectbox("Network", ["grid", "ring"], format_func=lambda x: {"grid": "Synthetic grid with river", "ring": "Synthetic ring-and-spoke"}[x])
    s = load_csv(str(config.REPORTS_DIR / "routing" / f"scenario_summary_{net}.csv"))
    if s is None:
        missing_artifact("Routing experiment results")
    else:
        st.subheader("Static fastest vs shortest vs traffic-aware routing")
        st.dataframe(s.round(2), hide_index=True, width="stretch")
        st.plotly_chart(px.bar(s, x="scenario", y="route_differs_from_static_%", color=s.severity.astype(str), barmode="group",
                               title="Trips where the traffic-aware route differs from the static fastest route"), width="stretch")
        st.plotly_chart(px.bar(s, x="scenario", y="mean_saved_vs_static_%", color=s.severity.astype(str), barmode="group",
                               title="Mean % time saved vs static fastest (trips with a feasible baseline)"), width="stretch")
        st.caption("'baseline_blocked_%' = trips where the no-traffic route would cross a closed road (excluded from time comparisons).")
        nz = load_csv(str(config.REPORTS_DIR / "routing" / f"noise_summary_{net}.csv"))
        if nz is not None:
            st.subheader("Forecast uncertainty: planning with noisy speeds, evaluating on the true scenario")
            st.dataframe(nz.round(2), hide_index=True, width="stretch")
            st.plotly_chart(px.line(nz, x="sigma", y=["mean_regret_pct", "mean_saved_vs_static_pct"], markers=True,
                                    title="Regret vs oracle and benefit vs static route as speed noise grows"), width="stretch")
    st.subheader("How scenarios change route selection for one trip")
    G = demo_graph("Synthetic grid with river (9x9)" if net == "grid" else "Synthetic ring-and-spoke")
    nodes = sorted(G.nodes); c = st.columns(3)
    o = c[0].selectbox("Origin node", nodes, index=0); d = c[1].selectbox("Destination node", nodes, index=len(nodes) - 1)
    sev = c[2].slider("Severity", 0.0, 1.0, 0.8, 0.05, key="exp_sev")
    if o != d:
        t = routing_evaluation.scenario_route_table(G, o, d, sev)
        if len(t):
            t["distinct_route_id"] = t.path.astype("category").cat.codes
            st.dataframe(t.drop(columns=["path_hash"]).round(2), hide_index=True, width="stretch")
            st.caption(f"{t.path.nunique()} distinct recommended paths across {len(t)} scenario x strategy combinations (same distinct_route_id = same path).")
    st.subheader("Real-world evaluation (predicted vs subsequently observed journey times)")
    up = st.file_uploader("Upload CSV with columns predicted_min, observed_min", type="csv", key="real_eval")
    if up:
        try:
            st.json(routing_evaluation.evaluate_predicted_vs_observed(pd.read_csv(up)))
        except Exception as e:
            st.error(str(e))
    else:
        st.info("No real evaluation data provided, so no real-world travel-time error or time-saving claim is made.")


def page_research():
    st.title("Research evaluation")
    m = load_csv(str(config.REPORTS_DIR / "forecast_metrics.csv")); cfg = load_json(str(config.REPORTS_DIR / "training_config.json"))
    if m is None:
        return missing_artifact("Forecast metrics")
    st.subheader("Forecasting metrics per junction and horizon")
    metric = st.radio("Metric", ["MAE", "RMSE", "R2"], horizontal=True)
    pv = m.pivot_table(index=["model", "junction"], columns="horizon", values=metric).round(3)
    st.dataframe(pv, width="stretch", height=420)
    st.subheader("Model vs baseline (MAE averaged over junctions)")
    t = m.groupby(["model", "horizon"]).MAE.mean().unstack().round(3); st.dataframe(t, width="stretch")
    st.subheader("Ablation: effect of calendar (temporal) features")
    ab = []
    for base, nc in (("Ridge", "Ridge_no_calendar"), ("RandomForest", "RandomForest_no_calendar"), ("LSTM_seed0", "LSTM_no_calendar_seed0")):
        if base in t.index and nc in t.index:
            for h in t.columns:
                ab.append({"model": base.replace("_seed0", ""), "horizon": h, "MAE with calendar": t.loc[base, h], "MAE without": t.loc[nc, h],
                           "change %": round(100 * (t.loc[nc, h] - t.loc[base, h]) / t.loc[nc, h], 1)})
    st.dataframe(pd.DataFrame(ab), hide_index=True, width="stretch")
    st.caption("Positive 'change %' = calendar features reduced MAE. LSTM ablation uses seed 0 only (single run).")
    st.subheader("Routing experiment results")
    for net in ("grid", "ring"):
        r = load_csv(str(config.REPORTS_DIR / "routing" / f"scenario_summary_{net}.csv"))
        if r is not None:
            st.write(f"**{net}** network (simulated)"); st.dataframe(r.round(2), hide_index=True, width="stretch")
    st.subheader("Reproducibility")
    if cfg:
        st.json({k: v for k, v in cfg.items() if k not in ("history", "congestion_thresholds")})
    rep = config.REPORTS_DIR / "research_report.md"
    if rep.exists():
        with st.expander("Full research report"):
            st.markdown(rep.read_text())
    st.subheader("Limitations")
    for l in (load_json(str(config.PROCESSED_DIR / "audit.json")) or {}).get("limitations", []):
        st.write("• " + l)
    st.write("• Routing results are simulated; no real journey-time validation exists in this project.")


def page_settings():
    st.title("Data sources & settings")
    ss = st.session_state
    st.subheader("Traffic-count dataset (Module A)")
    st.write(f"File: `{config.RAW_CSV}` - {'found' if config.RAW_CSV.exists() else '**missing** (download from Kaggle: fedesoriano/traffic-prediction-dataset)'}")
    st.write("Source: https://www.kaggle.com/datasets/fedesoriano/traffic-prediction-dataset (anonymous junctions, no geography).")
    st.subheader("Road network (Module B)")
    st.write(f"Current: **{ss.get('net_label', 'Synthetic demo (default)')}**. Graphs are cached in `{config.ROAD_CACHE_DIR}`. "
             "Load your own network on the Route planner page (GraphML upload or OSMnx download).")
    st.subheader("Forecast model")
    models = ["LSTM_ensemble", "RandomForest", "Ridge", "SeasonalNaive-168h"]
    st.selectbox("Model shown by default in forecasting page", models, key="forecast_model")
    st.caption("Junction forecasts are NOT connected to routing: they have no coordinates. See README 'Connecting forecasts to routing'.")
    st.subheader("Traffic input mode")
    st.write(f"Selected on the planner page: **{ss.get('traffic_mode', 'Synthetic scenario (demo)')}**")
    mode = ss.get("traffic_mode", "Synthetic scenario (demo)")
    st.info({"Synthetic": "SYNTHETIC - illustrative scenario speeds on a demo graph.", "Edge-speed": "OBSERVED/PREDICTED - depends on the uploaded file and its freshness.",
             "Assumed": "STATIC - assumed free-flow speeds only.", "Live": "LIVE API - unavailable unless an adapter and credentials are configured."}[mode.split()[0]])
    st.subheader("Optional authorised traffic API")
    st.write(f"TRAFFIC_API_KEY set: **{'yes' if os.environ.get('TRAFFIC_API_KEY') else 'no'}** (value never displayed). "
             f"ENABLE_GEOCODING: **{os.environ.get('ENABLE_GEOCODING', '0')}**. Credentials live in `.env` (see `.env.example`), never in code.")
    st.write("Internet is only needed for: OSM map tiles, OSMnx downloads, geocoding, and live APIs. Everything else runs offline.")


PAGE_FUNCS = [page_overview, page_analysis, page_forecast, page_planner, page_experiments, page_research, page_settings]


def main():
    try:
        from dotenv import load_dotenv; load_dotenv()
    except ImportError:
        pass
    page = st.sidebar.radio("Page", PAGES, key="page")
    st.sidebar.caption("Modules: A = junction count forecasting (Kaggle) · B = road-network routing (demo or your graph).")
    PAGE_FUNCS[PAGES.index(page)]()


main()
