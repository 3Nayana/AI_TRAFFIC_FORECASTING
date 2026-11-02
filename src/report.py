"""Generate reports/research_report.md. Every number is read from files produced by real runs."""
from __future__ import annotations

import json

import pandas as pd

from . import config

R = config.REPORTS_DIR


def _md(df: pd.DataFrame, nd: int = 2) -> str:
    return df.round(nd).to_markdown() if hasattr(df, "to_markdown") else df.round(nd).to_string()


def _t(df):
    try:
        return df.to_markdown()
    except ImportError:
        return "```\n" + df.to_string() + "\n```"


def write() -> None:
    audit = json.loads((config.PROCESSED_DIR / "audit.json").read_text())
    m = pd.read_csv(R / "forecast_metrics.csv"); cfg = json.loads((R / "training_config.json").read_text())
    mae = m.groupby(["model", "horizon"]).MAE.mean().unstack().round(2)
    rmse = m.groupby(["model", "horizon"]).RMSE.mean().unstack().round(2)
    r2 = m.groupby(["model", "horizon"]).R2.mean().unstack().round(3)
    per_j = m[m.model == "LSTM_ensemble"].pivot(index="junction", columns="horizon", values="MAE").round(2)
    per_j_b = m[m.model == "SeasonalNaive-168h"].pivot(index="junction", columns="horizon", values="MAE").round(2)
    seeds = m[m.model.str.match(r"LSTM_seed\d")].groupby(["model", "horizon"]).MAE.mean().unstack().round(2)
    ab = []
    for base, nc in (("Ridge", "Ridge_no_calendar"), ("RandomForest", "RandomForest_no_calendar"), ("LSTM_seed0", "LSTM_no_calendar_seed0")):
        for h in mae.columns:
            ab.append({"model": base.replace("_seed0", ""), "horizon": h, "MAE_with_calendar": mae.loc[base, h], "MAE_without": mae.loc[nc, h],
                       "improvement_%": round(100 * (mae.loc[nc, h] - mae.loc[base, h]) / mae.loc[nc, h], 1)})
    ab = pd.DataFrame(ab).pivot(index="model", columns="horizon", values="improvement_%")
    best = mae.drop(index=[i for i in mae.index if i.startswith("LSTM_seed") or "no_calendar" in i]).idxmin()
    routing = ""
    for net in ("grid", "ring"):
        f = R / "routing" / f"scenario_summary_{net}.csv"
        if f.exists():
            s = pd.read_csv(f).set_index(["scenario", "severity"])
            routing += f"\n**{net} network (simulated)**\n\n" + _t(s[["baseline_blocked_%", "route_differs_from_static_%", "faster_than_static_%",
                                                                    "faster_than_shortest_%", "mean_saved_vs_static_%"]].round(1)) + "\n"
    noise = ""
    f = R / "routing" / "noise_summary_grid.csv"
    if f.exists():
        noise = _t(pd.read_csv(f).round(2).set_index("sigma"))
    j = audit["junctions"]
    sn = mae.loc["SeasonalNaive-168h"]; ls = mae.loc["LSTM_ensemble"]
    lstm_vs = "; ".join(f"+{h}h: LSTM {ls[h]:.2f} vs seasonal-naive-168h {sn[h]:.2f} ({'LSTM better' if ls[h] < sn[h] else 'baseline better'})" for h in mae.columns)
    cal = "; ".join(f"{mname}: " + ", ".join(f"+{h}h {ab.loc[mname, h]:+.1f}%" for h in ab.columns) for mname in ab.index)
    answers = (f"* RQ1/H1 (calendar features): MAE change when adding calendar features - {cal}. Positive = helps. The effect is model- and horizon-dependent; "
               "where negative, calendar features did not help (LSTM ablation is single-seed, so treat it as indicative).\n"
               f"* H2 (LSTM vs seasonal naive): {lstm_vs}. The 168 h seasonal-naive baseline is strong because it carries the weekly pattern; the LSTM sees only the last "
               f"{cfg['window']} h, and training was capped at {cfg['epochs_max']} epochs on CPU, so it is likely under-tuned.\n")
    txt = f"""# AI Traffic Forecasting and Smart Route Recommendation - Research Report

*All numbers below were produced by running this repository (see `training_config.json`, `forecast_metrics.csv`, `routing/`).*

## 1. Objectives and research questions
Module A forecasts hourly vehicle counts per junction (LSTM as the main model, against baselines). Module B recommends alternative routes
on a road graph using edge travel times. The modules are joined only through a documented interface (`TrafficProvider`) that needs
**geographically matched edge speeds**; the Kaggle junctions have no coordinates, so no real link is claimed.

* RQ1 Do temporal features improve junction-count forecasts?
* RQ2 Do valid, geographically matched predictions improve travel-time estimates? **Not testable here** (no matched data).
* RQ3 Does traffic-aware routing reduce journey time vs shortest-distance / static-fastest? **Simulated only.**
* RQ4 How do route choices change with traffic intensity and forecast uncertainty? **Simulated only.**

Hypotheses: H1 calendar features reduce MAE; H2 an LSTM beats seasonal-naive at short horizons; H3 traffic-aware routing helps only when
congestion changes which route is fastest; H4 regret grows with speed noise.

## 2. Dataset (inspected from the supplied file)
{audit['rows']:,} rows, {audit['start'][:10]} to {audit['end'][:10]}, columns {audit['columns']}. Missing values: {audit['missing_values']}; duplicate rows: {audit['duplicate_rows']}.
All junctions are hourly with {sum(v['n_gaps'] for v in j.values())} gaps. Junction 4 has {j['4']['rows']:,} rows (starts {j['4']['start'][:10]}); the others {j['1']['rows']:,} rows.
Mean vehicles/hour: J1 {j['1']['vehicles']['mean']:.1f}, J2 {j['2']['vehicles']['mean']:.1f}, J3 {j['3']['vehicles']['mean']:.1f}, J4 {j['4']['vehicles']['mean']:.1f}.

Limitations: {'; '.join(audit['limitations'])}

## 3. Methods
**Split (chronological, by origin time):** train {cfg['splits']['train'][1]} and earlier; validation {cfg['splits']['val'][0]} to {cfg['splits']['val'][1]}; test {cfg['splits']['test'][0]} to {cfg['splits']['test'][1]}.
A sample's targets (up to +24 h) must lie inside its own split. Scalers and congestion thresholds use training data only.

**Features:** hour/weekday/month cyclical encodings, weekend flag (calendar of the *target* time, known in advance); lags 0,1,2,3,6,12,24,48,168 h and
rolling mean/std computed from past values only on a regular grid (gaps stay NaN, never bridged).
**Baselines:** persistence, seasonal-naive 24 h, seasonal-naive 168 h, historical mean by (weekday, hour).
**Models:** Ridge and RandomForest (one per junction and horizon); **LSTM** (2 layers, hidden {cfg['hidden']}, window {cfg['window']} h, one shared network for all junctions,
joint prediction of +1/+3/+6/+24 h, SmoothL1 loss, Adam, early stopping on validation loss, seeds {cfg['seeds']}; `LSTM_ensemble` = mean of seeds).
**Ablation:** same models without any calendar features.
**Congestion:** per-junction quantiles {cfg['congestion_quantiles']} of training counts -> Low/Moderate/High/Very high (relative to the junction's own history, not speed).

## 4. Forecasting results (test period, mean over 4 junctions; models compared on identical rows)
MAE (vehicles/hour):

{_t(mae)}

RMSE:

{_t(rmse)}

R2:

{_t(r2)}

Best non-ablation model per horizon by MAE: {best.to_dict()}.

LSTM MAE per seed (seed variability):

{_t(seeds)}

LSTM ensemble MAE per junction:

{_t(per_j)}

Seasonal-naive-168h MAE per junction:

{_t(per_j_b)}

### Ablation (RQ1): % MAE improvement from calendar features (positive = helps)
{_t(ab)}

## 5. Routing methodology
Directed graph (one-way preserved), edge time = length / (speed*1000/3600). Strategies: shortest distance; fastest baseline (assumed free-flow = 0.9 x limit);
lowest travel time; congestion-avoiding (t*(1+lambda*c)); balanced (normalised weighted time/distance/congestion). Candidates by Yen's k-shortest loopless paths with
edge-overlap de-duplication; the recommended route is the minimum of the declared objective over candidates. Closures remove edges. Time-dependent Dijkstra is used when
the provider varies with time. Scores: distance, time, avg speed, congested segments, exposure, turns, difference to baseline.

## 6. Routing experiments - SIMULATED, NOT real-world validation
Ground truth = the synthetic scenario speeds. Random origin-destination pairs (>=4 hops apart); a route counts as faster only if >0.5% faster.
{routing}

Forecast-uncertainty experiment (grid, heavy congestion severity 0.8; planner sees speeds x lognormal noise sigma, evaluated on true speeds):

{noise}

## 7. Answers (limited by the evidence)
{answers}* RQ2: untestable - no geographically matched traffic data was available; no claim is made.
* RQ3/RQ4: results above hold for synthetic scenarios on synthetic graphs only. They show when and how often routes change; they do not show real time savings.

## 8. Limitations
Unknown city/geography; counts are not speeds; 20 months only; single test window (May-Jun 2017); LSTM ablation uses one seed;
synthetic scenarios are hand-designed; no intersection delays; OSMnx download/geocoding/live API paths could not be exercised in the build sandbox (no internet).

## 9. Reproducibility
Python seeds {cfg['seeds']}, torch {cfg['torch']}, max {cfg['epochs_max']} epochs with early stopping, CPU only, run in resumable chunks. Run `python run_pipeline.py`; tests: `pytest`.
"""
    (R / "research_report.md").write_text(txt)


if __name__ == "__main__":
    write()
