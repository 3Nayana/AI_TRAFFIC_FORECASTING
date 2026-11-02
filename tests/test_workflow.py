"""End-to-end on a tiny synthetic fixture: data -> features -> models (incl. LSTM) -> predictions -> routing -> map."""
from datetime import datetime

import numpy as np
import pandas as pd

from src import data_loader, features, train, demo_network as dn
from src.alternative_routes import RouteConfig, Strategy, plan_routes
from src.evaluate import metrics_table
from src.map_view import build_map
from src.traffic_provider import ScenarioProvider
from tests.helpers import synthetic_traffic_csv


def test_forecasting_workflow_with_lstm_beats_nothing_broken():
    df = data_loader.validate(synthetic_traffic_csv(days=40)); grid = data_loader.to_hourly_grid(df)
    b = {"train": (pd.Timestamp.min, pd.Timestamp("2017-01-25 23:00")), "val": (pd.Timestamp("2017-01-26"), pd.Timestamp("2017-02-01 23:00")),
         "test": (pd.Timestamp("2017-02-02"), pd.Timestamp("2017-02-09 23:00"))}
    sc = features.Scaler.fit(grid, b["train"][1])
    H = (1, 3, 6, 24)
    tr, va, te = (features.sequence_samples(grid, sc, b[k], 24, H) for k in ("train", "val", "test"))
    model, hist = train.train_lstm(tr, va, seed=0, epochs=3, hidden=16)
    assert hist[-1]["train_loss"] < hist[0]["train_loss"]
    p = train.lstm_predict(model, te, sc); p["model"] = "LSTM"
    assert set(p.horizon) == set(H) and (p.pred >= 0).all() and np.isfinite(p.pred).all()
    m = metrics_table(p)
    assert {"MAE", "RMSE", "R2"} <= set(m.columns) and len(m) == 4 * 4
    base = train.baseline_predictions(grid, b["test"], b["train"][1])
    assert {"Persistence", "SeasonalNaive-168h", "SeasonalNaive-24h", "HistoricalMean(dow,hour)"} == set(base.model)
    tab = train.tabular_models(grid, b, True, "", save=False)
    assert {"Ridge", "RandomForest"} == set(tab.model)


def test_route_workflow_and_map_rendering(tmp_path):
    G = dn.build_grid_network(); prov = ScenarioProvider("heavy_congestion", 0.9)
    plan = plan_routes(G, 0, 80, prov, Strategy.CONGESTION_AVOID, RouteConfig(k=3), datetime(2026, 1, 12, 8, 0))
    m = build_map(G, prov.speeds(G, datetime(2026, 1, 12, 8, 0)), plan, selected=len(plan.candidates) - 1)
    f = tmp_path / "m.html"; m.save(str(f)); html = f.read_text()
    assert "Road congestion" in html and "Recommended route" in html and "Origin (snapped)" in html
    assert html.count("Route ") >= len(plan.candidates)
