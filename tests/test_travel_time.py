import math
import pytest
from src import travel_time as tt


def test_formula_known_value():
    assert tt.travel_time_seconds(1000, 36) == pytest.approx(100.0)       # 36 km/h = 10 m/s
    assert tt.travel_time_seconds(0, 50) == 0


@pytest.mark.parametrize("bad", [0, -5, float("nan"), float("inf"), None])
def test_invalid_speed_never_divides_by_zero(bad):
    with pytest.raises(ValueError):
        tt.travel_time_seconds(100, bad)


def test_sanitize_fallback_and_clamp():
    assert tt.sanitize_speed(None, 40) == (40.0, True)
    assert tt.sanitize_speed(0, 40) == (40.0, True)
    assert tt.sanitize_speed(1, 40) == (tt.MIN_SPEED_KMH, False)
    assert tt.sanitize_speed(500, 40) == (tt.MAX_SPEED_KMH, False)


def test_congestion_index_and_levels():
    assert tt.congestion_index(50, 50) == 0 and tt.congestion_index(0.1, 50) > 0.99
    assert tt.congestion_index(60, 50) == 0                      # faster than free-flow is not negative congestion
    assert [tt.congestion_level(c) for c in (0.0, 0.3, 0.5, 0.9)] == ["free", "light", "moderate", "heavy"]
