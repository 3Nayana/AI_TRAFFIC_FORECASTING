# AI Traffic Forecasting and Smart Route Recommendation System

Two connected but independent modules:

* **Module A - forecasting (LSTM + baselines).** Hourly vehicle counts for the four anonymous junctions in the Kaggle file
  `traffic.csv`, horizons +1/+3/+6/+24 h, relative congestion classes.
* **Module B - routing.** Directed road graph, alternative routes (Yen), five strategies, closures, time-dependent routing,
  Streamlit route planner. Runs **without** Module A and without any paid service.

**Honest scope.** The Kaggle data has no coordinates or road links and does not name a city. Therefore:
forecasts are *not* mapped to roads; the route planner uses a built-in demonstration network with clearly labelled
**synthetic** traffic scenarios. Synthetic speeds are illustrative; they are not observed, not predicted, and not outputs of the LSTM.
Simulated routing results are not real-world validation.

## Setup (Windows PowerShell)
```powershell
cd path\to\ai-traffic
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process Bypass
pip install -r requirements.txt       # osmnx/geopandas/xgboost are optional; remove them if install fails
mkdir data\raw -Force
# put traffic.csv in data\raw\  (Kaggle: fedesoriano/traffic-prediction-dataset; unzip archive.zip)
Copy-Item .env.example .env           # optional
```
macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate`.

## Run
```powershell
python run_pipeline.py --epochs 30    # audit+EDA -> train all models (LSTM 3 seeds + ablations) -> routing experiments -> report
python run_pipeline.py --skip-train   # reuse trained results
python run_pipeline.py --resume       # continue an interrupted training run
streamlit run app.py                  # dashboard (7 pages)
pytest                                # 90+ tests
```
Individual steps: `python -m src.eda`, `python -m src.train --quick`, `python -m src.routing_evaluation`, `python -m src.report`.
Training on a single CPU core took roughly 25-35 minutes. The Route Planner page works immediately, even before training.

## Route planner quick start
1. Page **4 Route planner**. Defaults: synthetic grid network, scenario *Heavy congestion*.
2. Choose "Map click sets: origin/destination" and click the map, or type coordinates. Press **Calculate alternative routes**.
3. Compare candidates (table, colours, radio selector). Change scenario/severity/strategy and recalculate - the route changes.
4. Map tiles are off by default (the demo coordinates are abstract; works offline). OpenStreetMap tiles, OSMnx download, geocoding and live APIs need internet.

### Strategies and objectives
| Strategy | Objective (per edge, summed over route) |
|---|---|
| Shortest distance | length |
| Fastest baseline | time at assumed free-flow speed (0.9 x limit) |
| Lowest travel time | time at the provider's speeds (scenario / observed / predicted) |
| Congestion-avoiding | t * (1 + lambda * c), c = clip(1 - v/v_ff, 0, 1) (deliberate extra risk penalty; lambda=0 = lowest time) |
| Balanced | w_t*t/T0 + w_d*len/D0 + w_c*(c*t)/T0, weights normalised |

If the data cannot support a strategy (e.g. no traffic data), the app **visibly** falls back (red FALLBACK banner) instead of pretending.

## Using a real road graph
```python
import osmnx as ox
G = ox.graph_from_place("Your place name", network_type="drive")   # or graph_from_point((lat, lon), dist=1500)
ox.save_graphml(G, "my_city.graphml")
```
Upload `my_city.graphml` on the planner page (Road network -> Upload GraphML), or use "Download via OSMnx". Graphs are cached in `data/road_cache/`.
The graph is standardised (directions, one-ways, lengths, geometry kept; parallel edges collapsed to the shortest) and reduced to its largest strongly connected component.
*Not exercised in my build sandbox (no internet): OSMnx download and geocoding. GraphML loading and standardisation are unit-tested.*

## Connecting forecasts to routing (documented interface)
Module B consumes `TrafficProvider.speeds(G, when) -> {edge_id: EdgeSpeed(speed_kmh, kind, closed, timestamp)}`.
To use real data you need **edge-level speeds matched to your graph**:
1. From an authorised source, collect point observations `lat, lon, speed_kmh, timestamp[, heading]`.
2. `traffic_provider.match_observations_to_edges` (nearest geometry within 30 m; ambiguous two-way roads need a heading) then `aggregate_matched`.
3. Save as CSV `edge_id,timestamp,speed_kmh[,closed]` and upload it on the planner page as *observed* or *predicted*. Freshness (max age), forecast tolerance, coverage and unit checks are enforced; failures fall back visibly.
4. Live APIs: implement `LiveAPIProvider.fetch()` for a vendor you are authorised to use; keys in `.env` only; follow its rate limits, caching and attribution terms.
Do **not** map anonymous Kaggle junction IDs to arbitrary edges.

## Project layout
`src/` data_loader, eda, features, train (baselines, Ridge, RF, LSTM), evaluate, congestion, road_network, demo_network, traffic_provider,
travel_time, alternative_routes, routing_evaluation, map_view, report - `app.py` - `tests/` - `reports/` (metrics, figures, routing CSVs, `research_report.md`).

## Known limitations
Single test window (May-Jun 2017); LSTM ablation single-seed; counts are not speeds; congestion classes are relative; no intersection delays modelled;
synthetic scenarios are hand-designed; map-click handling relies on `streamlit-folium` and was verified through Streamlit's test harness (pending-click logic) rather than a real browser.
