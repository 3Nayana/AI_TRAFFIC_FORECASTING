# AI Traffic Forecasting and Smart Route Recommendation - Research Report

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
48,120 rows, 2015-11-01 to 2017-06-30, columns ['DateTime', 'Junction', 'Vehicles', 'ID']. Missing values: {'DateTime': 0, 'Junction': 0, 'Vehicles': 0, 'ID': 0}; duplicate rows: 0.
All junctions are hourly with 0 gaps. Junction 4 has 4,344 rows (starts 2017-01-01); the others 14,592 rows.
Mean vehicles/hour: J1 45.1, J2 14.3, J3 13.7, J4 7.3.

Limitations: Anonymous junction IDs (1-4): no coordinates, road names, lane counts or capacities are provided, so counts cannot be placed on a map.; Vehicle COUNTS are not speeds, densities or travel times; congestion can only be expressed relatively.; Junction 4 starts on 2017-01-01 (about 6 months) while Junctions 1-3 span Nov 2015 - Jun 2017: unequal history and unequal train size.; About 20 months in total: yearly seasonality is barely observable; public holidays, weather and events are not recorded.; No vehicle classes, direction of travel or sensor-fault flags; outliers (e.g. the Junction 3 maximum) cannot be verified as real or sensor errors.; The data-collection city/country is not stated in the file; this project makes no geographic claim.

## 3. Methods
**Split (chronological, by origin time):** train 2017-02-28 23:00:00 and earlier; validation 2017-03-01 00:00:00 to 2017-04-30 23:00:00; test 2017-05-01 00:00:00 to 2017-06-30 23:00:00.
A sample's targets (up to +24 h) must lie inside its own split. Scalers and congestion thresholds use training data only.

**Features:** hour/weekday/month cyclical encodings, weekend flag (calendar of the *target* time, known in advance); lags 0,1,2,3,6,12,24,48,168 h and
rolling mean/std computed from past values only on a regular grid (gaps stay NaN, never bridged).
**Baselines:** persistence, seasonal-naive 24 h, seasonal-naive 168 h, historical mean by (weekday, hour).
**Models:** Ridge and RandomForest (one per junction and horizon); **LSTM** (2 layers, hidden 64, window 48 h, one shared network for all junctions,
joint prediction of +1/+3/+6/+24 h, SmoothL1 loss, Adam, early stopping on validation loss, seeds [0, 1, 2]; `LSTM_ensemble` = mean of seeds).
**Ablation:** same models without any calendar features.
**Congestion:** per-junction quantiles [0.5, 0.75, 0.9] of training counts -> Low/Moderate/High/Very high (relative to the junction's own history, not speed).

## 4. Forecasting results (test period, mean over 4 junctions; models compared on identical rows)
MAE (vehicles/hour):

| model                     |     1 |     3 |     6 |    24 |
|:--------------------------|------:|------:|------:|------:|
| HistoricalMean(dow,hour)  | 12.9  | 12.9  | 12.9  | 13.03 |
| LSTM_ensemble             |  3.75 |  4.65 |  5.28 |  6.36 |
| LSTM_no_calendar_ensemble |  3.16 |  4.03 |  5    |  6.32 |
| LSTM_no_calendar_seed0    |  3.16 |  4.03 |  5    |  6.32 |
| LSTM_seed0                |  4.25 |  5.26 |  6.01 |  7.13 |
| LSTM_seed1                |  3.71 |  4.72 |  5.35 |  6.6  |
| LSTM_seed2                |  3.7  |  4.45 |  4.92 |  5.87 |
| Persistence               |  3.93 |  7.27 | 11.58 |  6.28 |
| RandomForest              |  3.56 |  4.69 |  5.44 |  5.29 |
| RandomForest_no_calendar  |  3.78 |  5.59 |  6.95 |  6.55 |
| Ridge                     |  3.32 |  4.43 |  5.54 |  5.88 |
| Ridge_no_calendar         |  3.22 |  4.42 |  5.53 |  5.61 |
| SeasonalNaive-168h        |  4.03 |  4.03 |  4.03 |  3.98 |
| SeasonalNaive-24h         |  6.32 |  6.32 |  6.32 |  6.28 |

RMSE:

| model                     |     1 |     3 |     6 |    24 |
|:--------------------------|------:|------:|------:|------:|
| HistoricalMean(dow,hour)  | 15.2  | 15.2  | 15.2  | 15.33 |
| LSTM_ensemble             |  5.46 |  6.6  |  7.41 |  8.99 |
| LSTM_no_calendar_ensemble |  4.77 |  5.95 |  7.23 |  9.19 |
| LSTM_no_calendar_seed0    |  4.77 |  5.95 |  7.23 |  9.19 |
| LSTM_seed0                |  6.08 |  7.32 |  8.22 | 10.1  |
| LSTM_seed1                |  5.39 |  6.84 |  7.62 |  9.27 |
| LSTM_seed2                |  5.38 |  6.31 |  6.98 |  8.31 |
| Persistence               |  5.68 |  9.88 | 14.95 |  9.69 |
| RandomForest              |  5.43 |  6.99 |  7.91 |  7.68 |
| RandomForest_no_calendar  |  5.66 |  7.91 |  9.43 |  9.3  |
| Ridge                     |  4.9  |  6.26 |  7.44 |  7.81 |
| Ridge_no_calendar         |  4.85 |  6.37 |  7.58 |  7.82 |
| SeasonalNaive-168h        |  6.34 |  6.34 |  6.34 |  6.21 |
| SeasonalNaive-24h         |  9.73 |  9.73 |  9.73 |  9.69 |

R2:

| model                     |      1 |      3 |      6 |     24 |
|:--------------------------|-------:|-------:|-------:|-------:|
| HistoricalMean(dow,hour)  | -0.443 | -0.445 | -0.445 | -0.452 |
| LSTM_ensemble             |  0.702 |  0.62  |  0.563 |  0.431 |
| LSTM_no_calendar_ensemble |  0.727 |  0.639 |  0.535 |  0.39  |
| LSTM_no_calendar_seed0    |  0.727 |  0.639 |  0.535 |  0.39  |
| LSTM_seed0                |  0.663 |  0.566 |  0.499 |  0.329 |
| LSTM_seed1                |  0.712 |  0.61  |  0.553 |  0.408 |
| LSTM_seed2                |  0.704 |  0.639 |  0.589 |  0.493 |
| Persistence               |  0.62  |  0.22  | -0.554 |  0.252 |
| RandomForest              |  0.68  |  0.542 |  0.458 |  0.463 |
| RandomForest_no_calendar  |  0.651 |  0.452 |  0.259 |  0.335 |
| Ridge                     |  0.64  |  0.455 |  0.304 |  0.224 |
| Ridge_no_calendar         |  0.71  |  0.578 |  0.466 |  0.502 |
| SeasonalNaive-168h        |  0.444 |  0.444 |  0.443 |  0.464 |
| SeasonalNaive-24h         |  0.231 |  0.23  |  0.23  |  0.252 |

Best non-ablation model per horizon by MAE: {1: 'Ridge', 3: 'SeasonalNaive-168h', 6: 'SeasonalNaive-168h', 24: 'SeasonalNaive-168h'}.

LSTM MAE per seed (seed variability):

| model      |    1 |    3 |    6 |   24 |
|:-----------|-----:|-----:|-----:|-----:|
| LSTM_seed0 | 4.25 | 5.26 | 6.01 | 7.13 |
| LSTM_seed1 | 3.71 | 4.72 | 5.35 | 6.6  |
| LSTM_seed2 | 3.7  | 4.45 | 4.92 | 5.87 |

LSTM ensemble MAE per junction:

|   junction |    1 |    3 |     6 |    24 |
|-----------:|-----:|-----:|------:|------:|
|          1 | 6.93 | 9.55 | 11.51 | 14.03 |
|          2 | 3.54 | 4.02 |  4.17 |  5.46 |
|          3 | 2.68 | 3.08 |  3.45 |  3.91 |
|          4 | 1.87 | 1.94 |  1.98 |  2.04 |

Seasonal-naive-168h MAE per junction:

|   junction |    1 |    3 |    6 |   24 |
|-----------:|-----:|-----:|-----:|-----:|
|          1 | 5.77 | 5.77 | 5.78 | 5.59 |
|          2 | 3.01 | 3    | 3    | 2.96 |
|          3 | 4.67 | 4.66 | 4.66 | 4.65 |
|          4 | 2.68 | 2.69 | 2.69 | 2.7  |

### Ablation (RQ1): % MAE improvement from calendar features (positive = helps)
| model        |     1 |     3 |     6 |    24 |
|:-------------|------:|------:|------:|------:|
| LSTM         | -34.5 | -30.5 | -20.2 | -12.8 |
| RandomForest |   5.8 |  16.1 |  21.7 |  19.2 |
| Ridge        |  -3.1 |  -0.2 |  -0.2 |  -4.8 |

## 5. Routing methodology
Directed graph (one-way preserved), edge time = length / (speed*1000/3600). Strategies: shortest distance; fastest baseline (assumed free-flow = 0.9 x limit);
lowest travel time; congestion-avoiding (t*(1+lambda*c)); balanced (normalised weighted time/distance/congestion). Candidates by Yen's k-shortest loopless paths with
edge-overlap de-duplication; the recommended route is the minimum of the declared objective over candidates. Closures remove edges. Time-dependent Dijkstra is used when
the provider varies with time. Scores: distance, time, avg speed, congested segments, exposure, turns, difference to baseline.

## 6. Routing experiments - SIMULATED, NOT real-world validation
Ground truth = the synthetic scenario speeds. Random origin-destination pairs (>=4 hops apart); a route counts as faster only if >0.5% faster.

**grid network (simulated)**

|                           |   baseline_blocked_% |   route_differs_from_static_% |   faster_than_static_% |   faster_than_shortest_% |   mean_saved_vs_static_% |
|:--------------------------|---------------------:|------------------------------:|-----------------------:|-------------------------:|-------------------------:|
| ('daily_cycle', 0.3)      |                  0   |                           3.3 |                    3.3 |                     56.7 |                      0.1 |
| ('daily_cycle', 0.6)      |                  0   |                          23.3 |                   23.3 |                     45   |                      1.1 |
| ('daily_cycle', 0.9)      |                  0   |                          35   |                   31.7 |                     46.7 |                      4.6 |
| ('evening_peak', 0.3)     |                  0   |                           1.7 |                    0   |                     53.3 |                      0   |
| ('evening_peak', 0.6)     |                  0   |                           5   |                    5   |                     58.3 |                      0.2 |
| ('evening_peak', 0.9)     |                  0   |                          25   |                   25   |                     48.3 |                      1.5 |
| ('heavy_congestion', 0.3) |                  0   |                          21.7 |                   21.7 |                     45   |                      0.4 |
| ('heavy_congestion', 0.6) |                  0   |                          36.7 |                   33.3 |                     40   |                      3.8 |
| ('heavy_congestion', 0.9) |                  0   |                          48.3 |                   48.3 |                     46.7 |                     12.9 |
| ('incident_closure', 0.3) |                 13.3 |                           0   |                    0   |                     55   |                      0   |
| ('incident_closure', 0.6) |                 16.7 |                           0   |                    0   |                     55   |                      0   |
| ('incident_closure', 0.9) |                 16.7 |                          16   |                   14   |                     46.7 |                      0.4 |
| ('morning_peak', 0.3)     |                  0   |                           3.3 |                    1.7 |                     55   |                      0   |
| ('morning_peak', 0.6)     |                  0   |                           3.3 |                    3.3 |                     55   |                      0.2 |
| ('morning_peak', 0.9)     |                  0   |                          23.3 |                   23.3 |                     45   |                      1.7 |
| ('normal', 0.3)           |                  0   |                           0   |                    0   |                     53.3 |                      0   |
| ('normal', 0.6)           |                  0   |                           0   |                    0   |                     53.3 |                      0   |
| ('normal', 0.9)           |                  0   |                           0   |                    0   |                     53.3 |                      0   |

**ring network (simulated)**

|                           |   baseline_blocked_% |   route_differs_from_static_% |   faster_than_static_% |   faster_than_shortest_% |   mean_saved_vs_static_% |
|:--------------------------|---------------------:|------------------------------:|-----------------------:|-------------------------:|-------------------------:|
| ('daily_cycle', 0.3)      |                  0   |                          26.7 |                   11.7 |                     80   |                      0.3 |
| ('daily_cycle', 0.6)      |                  0   |                          65   |                   65   |                     46.7 |                      2.9 |
| ('daily_cycle', 0.9)      |                  0   |                          71.7 |                   68.3 |                     55   |                     11.2 |
| ('evening_peak', 0.3)     |                  0   |                          15   |                    6.7 |                     81.7 |                      0.1 |
| ('evening_peak', 0.6)     |                  0   |                          26.7 |                   26.7 |                     80   |                      0.9 |
| ('evening_peak', 0.9)     |                  0   |                          28.3 |                   26.7 |                     76.7 |                      2.4 |
| ('heavy_congestion', 0.3) |                  0   |                          55   |                   55   |                     51.7 |                      1.5 |
| ('heavy_congestion', 0.6) |                  0   |                          70   |                   70   |                     43.3 |                      8.7 |
| ('heavy_congestion', 0.9) |                  0   |                          85   |                   85   |                     66.7 |                     23.2 |
| ('incident_closure', 0.3) |                 36.7 |                          10.5 |                   10.5 |                     81.7 |                      0.1 |
| ('incident_closure', 0.6) |                 68.3 |                          21.1 |                   21.1 |                     70   |                      1.3 |
| ('incident_closure', 0.9) |                 68.3 |                          26.3 |                   26.3 |                     70   |                      2.7 |
| ('morning_peak', 0.3)     |                  0   |                          11.7 |                    1.7 |                     86.7 |                      0   |
| ('morning_peak', 0.6)     |                  0   |                          25   |                   25   |                     83.3 |                      0.8 |
| ('morning_peak', 0.9)     |                  0   |                          30   |                   28.3 |                     76.7 |                      2.3 |
| ('normal', 0.3)           |                  0   |                           1.7 |                    1.7 |                     88.3 |                      0   |
| ('normal', 0.6)           |                  0   |                           1.7 |                    1.7 |                     88.3 |                      0   |
| ('normal', 0.9)           |                  0   |                           1.7 |                    1.7 |                     88.3 |                      0   |


Forecast-uncertainty experiment (grid, heavy congestion severity 0.8; planner sees speeds x lognormal noise sigma, evaluated on true speeds):

|   sigma |   trips |   mean_regret_pct |   pct_same_as_oracle |   mean_saved_vs_static_pct |
|--------:|--------:|------------------:|---------------------:|---------------------------:|
|     0   |      30 |              0    |               100    |                      11.83 |
|     0.1 |      90 |              0.16 |                88.89 |                      11.68 |
|     0.2 |      90 |              1.24 |                78.89 |                      10.7  |
|     0.4 |      90 |              8.04 |                48.89 |                       5.06 |
|     0.8 |      90 |             24.35 |                25.56 |                      -8.44 |

## 7. Answers (limited by the evidence)
* RQ1/H1 (calendar features): MAE change when adding calendar features - LSTM: +1h -34.5%, +3h -30.5%, +6h -20.2%, +24h -12.8%; RandomForest: +1h +5.8%, +3h +16.1%, +6h +21.7%, +24h +19.2%; Ridge: +1h -3.1%, +3h -0.2%, +6h -0.2%, +24h -4.8%. Positive = helps. The effect is model- and horizon-dependent; where negative, calendar features did not help (LSTM ablation is single-seed, so treat it as indicative).
* H2 (LSTM vs seasonal naive): +1h: LSTM 3.75 vs seasonal-naive-168h 4.03 (LSTM better); +3h: LSTM 4.65 vs seasonal-naive-168h 4.03 (baseline better); +6h: LSTM 5.28 vs seasonal-naive-168h 4.03 (baseline better); +24h: LSTM 6.36 vs seasonal-naive-168h 3.98 (baseline better). The 168 h seasonal-naive baseline is strong because it carries the weekly pattern; the LSTM sees only the last 48 h, and training was capped at 12 epochs on CPU, so it is likely under-tuned.
* RQ2: untestable - no geographically matched traffic data was available; no claim is made.
* RQ3/RQ4: results above hold for synthetic scenarios on synthetic graphs only. They show when and how often routes change; they do not show real time savings.

## 8. Limitations
Unknown city/geography; counts are not speeds; 20 months only; single test window (May-Jun 2017); LSTM ablation uses one seed;
synthetic scenarios are hand-designed; no intersection delays; OSMnx download/geocoding/live API paths could not be exercised in the build sandbox (no internet).

## 9. Reproducibility
Python seeds [0, 1, 2], torch 2.14.1+cu130, max 12 epochs with early stopping, CPU only, run in resumable chunks. Run `python run_pipeline.py`; tests: `pytest`.
