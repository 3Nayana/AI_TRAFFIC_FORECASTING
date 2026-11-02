import numpy as np
import pandas as pd
import pytest

from src import congestion, data_loader, features
from tests.helpers import synthetic_traffic_csv


@pytest.fixture(scope="module")
def df():
    return data_loader.validate(synthetic_traffic_csv())


@pytest.fixture(scope="module")
def grid(df):
    return data_loader.to_hourly_grid(df)


# ---------------- dataset validation
def test_validate_ok(df):
    assert list(df.columns) == ["DateTime", "Junction", "Vehicles", "ID"]
    assert str(df.Junction.dtype).startswith("int") and df.Vehicles.dtype == float


def test_validate_rejects_missing_column():
    with pytest.raises(data_loader.DataValidationError):
        data_loader.validate(pd.DataFrame({"DateTime": ["2017-01-01"], "Junction": [1], "Vehicles": [3]}))


def test_validate_rejects_bad_values():
    bad = synthetic_traffic_csv(days=2)
    with pytest.raises(data_loader.DataValidationError):
        data_loader.validate(bad.assign(Vehicles=-1))
    with pytest.raises(data_loader.DataValidationError):
        data_loader.validate(bad.assign(DateTime="not a date"))


def test_audit_detects_gap_and_duplicates(df):
    d = df[~((df.Junction == 1) & (df.DateTime.between("2017-01-05 00:00", "2017-01-05 05:00")))]
    d = pd.concat([d, d.iloc[:3]])
    rep = data_loader.audit(d)
    assert rep["duplicate_rows"] == 3
    assert rep["junctions"][1]["missing_timestamps"] == 6 and rep["junctions"][1]["n_gaps"] == 1


def test_missing_timestamps_become_nan_not_consecutive(df):
    d = df[~((df.Junction == 1) & (df.DateTime.between("2017-01-05 00:00", "2017-01-05 05:00")))]
    g = data_loader.to_hourly_grid(d)
    assert g.loc["2017-01-05 00:00":"2017-01-05 05:00", 1].isna().all()
    assert len(g) == len(data_loader.to_hourly_grid(df))   # grid length unchanged -> hours not collapsed


@pytest.mark.skipif(not data_loader.config.RAW_CSV.exists(), reason="traffic.csv not present")
def test_real_file_schema():
    rep = data_loader.audit(data_loader.load_raw())
    assert rep["rows"] > 0 and set(rep["junctions"]) == {1, 2, 3, 4}
    assert rep["duplicate_junction_timestamp"] == 0


# ---------------- leakage and splits
def test_past_only_features_do_not_see_the_future(grid):
    s = grid[1]; t0 = s.index[500]
    f1 = features.past_only_features(s)
    s2 = s.copy(); s2.loc[s2.index > t0] += 1000          # corrupt every future value
    f2 = features.past_only_features(s2)
    pd.testing.assert_frame_equal(f1.loc[:t0], f2.loc[:t0])


def test_target_alignment_and_calendar_is_target_time(grid):
    fr = features.tabular_frame(grid, 1, 3, True)
    t = fr.index[10]
    assert fr.loc[t, "y"] == grid.loc[t + pd.Timedelta(hours=3), 1]
    expect = np.sin(2 * np.pi * (t + pd.Timedelta(hours=3)).hour / 24)
    assert fr.loc[t, "tgt_hour_sin1"] == pytest.approx(expect)
    assert fr.loc[t, "lag_0"] == grid.loc[t, 1]


def test_gap_makes_lags_nan_instead_of_shifting(df):
    d = df[~((df.Junction == 1) & (df.DateTime.between("2017-01-10 00:00", "2017-01-10 05:00")))]
    g = data_loader.to_hourly_grid(d)
    f = features.past_only_features(g[1])
    assert np.isnan(f.loc["2017-01-10 06:00", "lag_1"])
    assert np.isnan(f.loc["2017-01-10 06:00", "roll_mean_24"])      # strict rolling window
    assert not np.isnan(f.loc["2017-01-12 12:00", "lag_1"])


def test_sequence_windows_never_bridge_gaps(df):
    d = df[~((df.Junction == 2) & (df.DateTime.between("2017-01-20 00:00", "2017-01-20 02:00")))]
    g = data_loader.to_hourly_grid(d)
    sc = features.Scaler.fit(g, pd.Timestamp("2017-02-10"))
    b = (pd.Timestamp("2017-01-01"), pd.Timestamp("2017-02-28 23:00"))
    s = features.sequence_samples(g, sc, b, window=24, horizons=(1, 24))
    bad = (s["junction"] == 2) & (pd.to_datetime(s["origin"]) >= "2017-01-19 23:00") & (pd.to_datetime(s["origin"]) < "2017-01-21 02:00")
    # windows touching the gap (as past) or targets inside the gap are excluded
    assert not bad.any()
    assert not np.isnan(s["X"]).any() and not np.isnan(s["y"]).any()


def test_chronological_split_no_overlap_and_targets_inside(grid):
    b = {"train": (pd.Timestamp.min, pd.Timestamp("2017-01-31 23:00")),
         "val": (pd.Timestamp("2017-02-01"), pd.Timestamp("2017-02-14 23:00")),
         "test": (pd.Timestamp("2017-02-15"), pd.Timestamp("2017-02-28 23:00"))}
    sc = features.Scaler.fit(grid, b["train"][1])
    parts = {k: features.sequence_samples(grid, sc, v, 24, (1, 3, 6, 24)) for k, v in b.items()}
    o = {k: pd.to_datetime(p["origin"]) for k, p in parts.items()}
    assert o["train"].max() < o["val"].min() and o["val"].max() < o["test"].min()
    for k, (lo, hi) in b.items():          # every target (origin+24h) stays inside its own split
        assert (o[k] + pd.Timedelta(hours=24) <= hi).all()


def test_scaler_uses_training_period_only(grid):
    sc1 = features.Scaler.fit(grid, pd.Timestamp("2017-01-31 23:00"))
    g2 = grid.copy(); g2.loc["2017-02-01":] *= 100
    sc2 = features.Scaler.fit(g2, pd.Timestamp("2017-01-31 23:00"))
    assert sc1.mean == sc2.mean and sc1.std == sc2.std


# ---------------- congestion
def test_congestion_thresholds_train_only_and_monotone(grid):
    end = pd.Timestamp("2017-01-31 23:00")
    t1 = congestion.fit_thresholds(grid, end)
    g2 = grid.copy(); g2.loc["2017-02-01":] *= 50
    assert t1 == congestion.fit_thresholds(g2, end)
    th = t1[1]["all"]; assert th == sorted(th)
    lab = congestion.classify([0, th[0] + 1e-9, th[1] + 1e-9, th[2] + 1e-9, 1e9], 1, [0] * 5, t1)
    assert list(lab[:4]) == ["Low", "Moderate", "High", "Very high"] and lab[-1] == "Very high"
