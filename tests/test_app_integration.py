"""Streamlit integration: UI controls -> route engine -> results. Uses Streamlit's AppTest (no browser)."""
import pytest
from streamlit.testing.v1 import AppTest

from pathlib import Path
APP = str(Path(__file__).resolve().parents[1] / "app.py")


def _planner():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.sidebar.radio[0].set_value("4 Route planner").run()
    assert not at.exception
    return at


def _calc(at):
    btn = [b for b in at.button if "Calculate" in b.label][0]
    return btn.click().run()


def test_all_pages_render_without_exceptions():
    at = AppTest.from_file(APP, default_timeout=120).run()
    for p in at.sidebar.radio[0].options:
        at.sidebar.radio[0].set_value(p).run()
        assert not at.exception, (p, [e.value for e in at.exception])


def test_calculate_routes_default_demo():
    at = _calc(_planner())
    assert not at.exception
    plan = at.session_state["plan"]
    assert plan is not None and 1 <= len(plan.candidates) <= 3
    assert plan.status.synthetic and any("SYNTHETIC" in w for w in plan.warnings)
    assert any("Recommended" in s.value for s in at.success)


@pytest.mark.parametrize("strategy_idx", range(5))
def test_every_strategy_is_wired(strategy_idx):
    at = _planner()
    sb = [s for s in at.selectbox if s.label == "Routing strategy"][0]
    sb.set_value(sb.options[strategy_idx]); at.run()
    at = _calc(at)
    assert not at.exception
    plan = at.session_state["plan"]
    assert plan is not None and plan.candidates[0].is_recommended
    key = plan.strategy_used.value
    assert plan.candidates[0].objective[key] == min(c.objective[key] for c in plan.candidates)


def test_scenario_and_severity_controls_change_the_result():
    at = _planner()
    sc = [s for s in at.selectbox if s.label == "Scenario"][0]
    sev = [s for s in at.slider if s.label == "Congestion severity"][0]
    at.number_input(key="o_lat").set_value(at.session_state["o_lat"]); 
    sc.set_value("normal"); sev.set_value(0.0); at.run(); at = _calc(at)
    t_norm = at.session_state["plan"].candidates[0].time_min
    [s for s in at.selectbox if s.label == "Scenario"][0].set_value("heavy_congestion")
    [s for s in at.slider if s.label == "Congestion severity"][0].set_value(1.0); at.run(); at = _calc(at)
    assert at.session_state["plan"].candidates[0].time_min > t_norm


def test_alternatives_slider_and_comparison_table():
    at = _planner()
    [s for s in at.slider if s.label == "Number of alternatives"][0].set_value(4); at.run(); at = _calc(at)
    assert len(at.session_state["plan"].candidates) == 4
    assert len(at.dataframe) >= 1


def test_invalid_selection_shows_helpful_error():
    at = _planner()
    at.number_input(key="o_lat").set_value(10.0).run(); at.number_input(key="o_lon").set_value(10.0).run()
    at = _calc(at)
    assert not at.exception and at.session_state["plan"] is None
    assert any("outside network coverage" in e.value for e in at.error)


def test_live_mode_falls_back_visibly():
    at = _planner()
    [s for s in at.selectbox if s.label == "Traffic input mode"][0].set_value("Live traffic API").run(); at = _calc(at)
    plan = at.session_state["plan"]
    assert plan is not None and plan.fallbacks and any("FALLBACK" in e.value for e in at.error)


def test_map_click_is_applied_to_origin_coordinates():
    at = _planner()
    at.session_state["_pending_click"] = {"mode": "Set origin", "lat": 0.0012, "lng": -0.0011}
    at.run()
    assert at.session_state["o_lat"] == 0.0012 and at.session_state["o_lon"] == -0.0011
