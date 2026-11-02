"""Train and evaluate forecasting models: baselines, Ridge, RandomForest, LSTM (+ ablations).

Usage:  python -m src.train [--epochs 40] [--seeds 0 1 2] [--quick]
Everything saved here is produced by actually running the experiments.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import json
import logging
import time
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from . import config, congestion, data_loader, evaluate, features

log = logging.getLogger(__name__)
torch.set_num_threads(max(1, __import__("os").cpu_count() or 1))


# ----------------------------------------------------------------------------- stage cache (resumable runs)
STAGE_DIR = config.PROCESSED_DIR / "stage_cache"


def cached(name: str, fn, resume: bool):
    """Run fn() -> (DataFrame, extra_json) or load from disk when resume=True and the stage already finished."""
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    f, j = STAGE_DIR / f"{name}.csv.gz", STAGE_DIR / f"{name}.json"
    if resume and f.exists() and j.exists():
        log.info("resuming: loaded cached stage %s", name)
        return pd.read_csv(f, parse_dates=["origin_time", "target_time"]), json.loads(j.read_text())
    df, extra = fn()
    df.to_csv(f, index=False); j.write_text(json.dumps(extra))
    return df, extra


# ----------------------------------------------------------------------------- baselines
def baseline_predictions(grid: pd.DataFrame, bounds, thresholds_train_end: pd.Timestamp) -> pd.DataFrame:
    H = max(config.HORIZONS)
    tr = grid.loc[:thresholds_train_end]
    rows: List[pd.DataFrame] = []
    for j in grid.columns:
        s = grid[j]
        hist = tr[j].dropna()
        prof = hist.groupby([hist.index.dayofweek, hist.index.hour]).mean()
        for h in config.HORIZONS:
            tgt_time = grid.index + pd.Timedelta(hours=h)
            k = int(np.ceil(h / 24))
            preds = {
                "Persistence": s,
                "SeasonalNaive-24h": s.shift(24 * k - h),
                "SeasonalNaive-168h": s.shift(168 - h),
                "HistoricalMean(dow,hour)": pd.Series(
                    [prof.get((d, hr), np.nan) for d, hr in zip(tgt_time.dayofweek, tgt_time.hour)], index=grid.index),
            }
            y = s.shift(-h)
            ok = features.in_split(grid.index, bounds, H)
            for m, p in preds.items():
                df = pd.DataFrame({"origin_time": grid.index, "target_time": tgt_time, "junction": int(j), "horizon": h,
                                   "model": m, "actual": y.values, "pred": p.values})[ok]
                rows.append(df.dropna(subset=["actual", "pred"]))
    return pd.concat(rows, ignore_index=True)


# ----------------------------------------------------------------------------- tabular models
def tabular_models(grid: pd.DataFrame, bounds: Dict, use_calendar: bool, tag: str, save: bool = True) -> pd.DataFrame:
    H = max(config.HORIZONS); rows = []
    for j in grid.columns:
        for h in config.HORIZONS:
            fr = features.tabular_frame(grid, int(j), h, use_calendar)
            orig = fr.index
            tr = fr[features.in_split(orig, bounds["train"], H)]
            te = fr[features.in_split(orig, bounds["test"], H)]
            cols = [c for c in fr.columns if c not in ("y", "target_time")]
            models = {
                f"Ridge{tag}": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
                f"RandomForest{tag}": RandomForestRegressor(n_estimators=100, min_samples_leaf=5, max_features=0.5,
                                                          n_jobs=-1, random_state=0),
            }
            for name, mdl in models.items():
                mdl.fit(tr[cols].values, tr["y"].values)
                p = np.clip(mdl.predict(te[cols].values), 0, None)
                rows.append(pd.DataFrame({"origin_time": te.index, "target_time": te["target_time"].values,
                                          "junction": int(j), "horizon": h, "model": name,
                                          "actual": te["y"].values, "pred": p}))
                if save and tag == "":
                    joblib.dump(mdl, config.MODELS_DIR / f"{name}_j{j}_h{h}.joblib", compress=3)
    return pd.concat(rows, ignore_index=True)


# ----------------------------------------------------------------------------- LSTM
class TrafficLSTM(nn.Module):
    """LSTM encoder over the past window + MLP head that sees the junction one-hot and the
    (known-in-advance) calendar features of every target time; predicts all horizons jointly."""

    def __init__(self, in_dim: int, extra_dim: int, n_out: int, hidden: int = 64, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden + extra_dim, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, n_out))

    def forward(self, x, extra):
        out, _ = self.lstm(x)
        return self.head(torch.cat([out[:, -1], extra], dim=1))


def _batches(n: int, bs: int, shuffle: bool, rng: np.random.Generator):
    idx = rng.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


class TimeBudgetExceeded(SystemExit):
    """Raised (as a clean exit) when TRAIN_BUDGET_S is hit; progress is checkpointed, rerun with --resume."""


def train_lstm(tr: Dict, va: Dict, seed: int, epochs: int, hidden: int, patience: int = 6, bs: int = 256, lr: float = 2e-3,
               ckpt: "Path | None" = None):
    config.set_seed(seed)
    rng = np.random.default_rng(seed)
    model = TrafficLSTM(tr["X"].shape[2], tr["extra"].shape[1], tr["y"].shape[1], hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    lossf = nn.SmoothL1Loss()
    T = {k: torch.from_numpy(v) for k, v in tr.items() if k in ("X", "extra", "y")}
    V = {k: torch.from_numpy(v) for k, v in va.items() if k in ("X", "extra", "y")}
    best, best_state, bad, hist, start = np.inf, None, 0, [], 0
    if ckpt is not None and Path(ckpt).exists():           # resume an interrupted run (same seed/config)
        c = torch.load(ckpt, weights_only=False)
        model.load_state_dict(c["model"]); opt.load_state_dict(c["opt"]); sched.load_state_dict(c["sched"])
        best, best_state, bad, hist, start = c["best"], c["best_state"], c["bad"], c["hist"], c["epoch"]
        rng = np.random.default_rng(seed + 1000 * start)
        log.info("resumed %s at epoch %d", ckpt, start)
    budget = float(os.environ.get("TRAIN_BUDGET_S", "0") or 0); t_begin = time.time()
    for ep in range(start, epochs):
        if bad >= patience:
            break
        if budget and ckpt is not None and time.time() - t_begin + 25 > budget and ep > start:
            raise TimeBudgetExceeded(3)
        model.train(); t0 = time.time(); tl = 0.0
        for b in _batches(len(T["y"]), bs, True, rng):
            opt.zero_grad(); loss = lossf(model(T["X"][b], T["extra"][b]), T["y"][b]); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); tl += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            vl = float(np.mean([lossf(model(V["X"][b], V["extra"][b]), V["y"][b]).item() for b in _batches(len(V["y"]), 1024, False, rng)]))
        sched.step(vl); hist.append({"epoch": ep + 1, "train_loss": tl / len(T["y"]), "val_loss": vl, "sec": time.time() - t0})
        log.info("seed %d epoch %02d train %.4f val %.4f (%.1fs)", seed, ep + 1, hist[-1]["train_loss"], vl, hist[-1]["sec"])
        if vl < best - 1e-5:
            best, bad = vl, 0; best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if ckpt is not None:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "best": best,
                        "best_state": best_state, "bad": bad, "hist": hist, "epoch": ep + 1}, ckpt)
    model.load_state_dict(best_state)
    return model, hist


def lstm_predict(model, d: Dict, scaler: features.Scaler) -> pd.DataFrame:
    model.eval(); outs = []
    with torch.no_grad():
        for b in _batches(len(d["y"]), 1024, False, None):
            outs.append(model(torch.from_numpy(d["X"][b]), torch.from_numpy(d["extra"][b])).numpy())
    z = np.concatenate(outs)
    rows = []
    for hi, h in enumerate(config.HORIZONS):
        pred = np.array([scaler.inv(int(j), v) for j, v in zip(d["junction"], z[:, hi])])
        origin = pd.to_datetime(d["origin"])
        rows.append(pd.DataFrame({"origin_time": origin, "target_time": origin + pd.Timedelta(hours=h),
                                  "junction": d["junction"], "horizon": h, "actual": d["y_raw"][:, hi],
                                  "pred": np.clip(pred, 0, None)}))
    return pd.concat(rows, ignore_index=True)


def lstm_experiment(grid, scaler, bounds, use_calendar: bool, name: str, seeds, epochs: int, hidden: int, save: bool, resume: bool = False):
    tr = features.sequence_samples(grid, scaler, bounds["train"], config.LSTM_WINDOW, config.HORIZONS, use_calendar)
    va = features.sequence_samples(grid, scaler, bounds["val"], config.LSTM_WINDOW, config.HORIZONS, use_calendar)
    te = features.sequence_samples(grid, scaler, bounds["test"], config.LSTM_WINDOW, config.HORIZONS, use_calendar)
    log.info("%s samples: train %d val %d test %d", name, len(tr["y"]), len(va["y"]), len(te["y"]))
    preds, histories = [], {}
    for s in seeds:
        def run_seed():
            STAGE_DIR.mkdir(parents=True, exist_ok=True)
            model, hist = train_lstm(tr, va, s, epochs, hidden, ckpt=STAGE_DIR / f"ckpt_{name}_seed{s}.pt")
            p = lstm_predict(model, te, scaler); p["model"] = f"{name}_seed{s}"
            if save:
                torch.save({"state": model.state_dict(), "in_dim": tr["X"].shape[2], "extra_dim": tr["extra"].shape[1],
                            "hidden": hidden, "window": config.LSTM_WINDOW, "use_calendar": use_calendar,
                            "scaler": {"mean": scaler.mean, "std": scaler.std}}, config.MODELS_DIR / f"{name}_seed{s}.pt")
            return p, hist
        p, hist = cached(f"{name}_seed{s}", run_seed, resume)
        preds.append(p); histories[s] = hist
    allp = pd.concat(preds, ignore_index=True)
    ens = (allp.groupby(["origin_time", "target_time", "junction", "horizon"], as_index=False)
           .agg(actual=("actual", "first"), pred=("pred", "mean")))
    ens["model"] = f"{name}_ensemble"
    return pd.concat([allp, ens], ignore_index=True), histories


def main(epochs: int = 40, seeds=config.SEEDS, hidden: int = 64, ablation_seeds=(0,), save: bool = True, resume: bool = False) -> None:
    config.setup_logging(); config.MODELS_DIR.mkdir(exist_ok=True, parents=True)
    t_start = time.time()
    df = data_loader.load_raw(); grid = data_loader.to_hourly_grid(df)
    bounds = data_loader.split_bounds()
    scaler = features.Scaler.fit(grid, bounds["train"][1])
    parts = [cached("baselines", lambda: (baseline_predictions(grid, bounds["test"], bounds["train"][1]), {}), resume)[0]]
    log.info("baselines done")
    parts.append(cached("tabular_calendar", lambda: (tabular_models(grid, bounds, True, "", save), {}), resume)[0]); log.info("tabular (calendar) done")
    parts.append(cached("tabular_no_calendar", lambda: (tabular_models(grid, bounds, False, "_no_calendar", save), {}), resume)[0]); log.info("tabular (no calendar) done")
    lp, hist = lstm_experiment(grid, scaler, bounds, True, "LSTM", seeds, epochs, hidden, save, resume); parts.append(lp)
    lp2, hist2 = lstm_experiment(grid, scaler, bounds, False, "LSTM_no_calendar", ablation_seeds, epochs, hidden, save, resume); parts.append(lp2)
    pred = pd.concat(parts, ignore_index=True)

    # Compare every model on exactly the same test rows.
    key = ["junction", "origin_time", "horizon"]
    counts = pred.groupby(key).model.nunique()
    full = counts[counts == pred.model.nunique()].index
    common = pred.set_index(key).loc[full].reset_index()
    log.info("common test rows: %d (of %d)", len(common) // pred.model.nunique(), len(pred) // pred.model.nunique())
    metrics = evaluate.metrics_table(common)
    metrics = evaluate.add_skill(metrics, "SeasonalNaive-168h")
    common.to_csv(config.PROCESSED_DIR / "predictions_test.csv.gz", index=False)
    metrics.to_csv(config.REPORTS_DIR / "forecast_metrics.csv", index=False)
    thr = congestion.fit_thresholds(grid, bounds["train"][1])
    cfg = {"horizons": config.HORIZONS, "window": config.LSTM_WINDOW, "seeds": list(seeds), "ablation_seeds": list(ablation_seeds),
           "epochs_max": epochs, "hidden": hidden, "splits": {k: [str(a), str(b)] for k, (a, b) in bounds.items()},
           "congestion_quantiles": config.CONGESTION_QUANTILES, "congestion_thresholds": thr,
           "scaler": {"mean": scaler.mean, "std": scaler.std},
           "torch": torch.__version__, "train_minutes": round((time.time() - t_start) / 60, 2),
           "history": {"LSTM": {str(k): v for k, v in hist.items()}, "LSTM_no_calendar": {str(k): v for k, v in hist2.items()}}}
    (config.REPORTS_DIR / "training_config.json").write_text(json.dumps(cfg, indent=2, default=str))
    # plots
    for j in grid.columns:
        evaluate.plot_actual_vs_pred(common, int(j), 1, ["SeasonalNaive-168h", "RandomForest", "LSTM_ensemble"], config.FIGURES_DIR / f"forecast_j{j}_h1.png")
        evaluate.plot_actual_vs_pred(common, int(j), 24, ["SeasonalNaive-168h", "RandomForest", "LSTM_ensemble"], config.FIGURES_DIR / f"forecast_j{j}_h24.png")
    log.info("training finished in %.1f min", (time.time() - t_start) / 60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(config.SEEDS))
    ap.add_argument("--quick", action="store_true", help="2 epochs, 1 seed (smoke test)")
    ap.add_argument("--resume", action="store_true", help="reuse finished stages from data/processed/stage_cache")
    a = ap.parse_args()
    if a.quick:
        main(epochs=2, seeds=(0,), hidden=a.hidden)
    else:
        main(a.epochs, tuple(a.seeds), a.hidden, resume=a.resume)
