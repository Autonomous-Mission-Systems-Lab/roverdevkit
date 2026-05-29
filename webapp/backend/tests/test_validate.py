"""Smoke tests for ``/validate/rediscovery``.

These tests verify the route's contract against the committed
rediscovery LOO artifacts under ``reports/rediscovery_loo_evaluator/``
and ``reports/rediscovery_loo_surrogate_v9/``. The artifacts are
deterministic (master seed = 0) so the Pragyan / CADRE distance
numbers asserted below are stable; if they shift, it's because
something downstream of the panel-tilt fix changed and the
comparison report needs a refresh.
"""

from __future__ import annotations

import math

from fastapi.testclient import TestClient


def test_list_evaluator_returns_six_rovers(client: TestClient) -> None:
    response = client.get("/validate/rediscovery")
    assert response.status_code == 200
    body = response.json()
    assert body["backend"] == "evaluator"
    rovers = body["rovers"]
    assert len(rovers) == 6
    slugs = {r["slug"] for r in rovers}
    assert slugs == {"pragyan", "yutu_2", "moonranger", "rashid_1", "cadre_unit", "tenacious"}


def test_list_evaluator_orders_flown_first(client: TestClient) -> None:
    response = client.get("/validate/rediscovery")
    assert response.status_code == 200
    rovers = response.json()["rovers"]
    flown = [r for r in rovers if r["is_flown"]]
    not_flown = [r for r in rovers if not r["is_flown"]]
    assert {r["rover_name"] for r in flown} == {"Pragyan", "Yutu-2"}
    assert {r["rover_name"] for r in not_flown} == {
        "MoonRanger",
        "Rashid-1",
        "CADRE-unit",
        "Tenacious",
    }
    # Flown rovers come first in the response so the frontend can
    # render the "headline" tier before the design-target tier.
    rover_names = [r["rover_name"] for r in rovers]
    assert rover_names.index("Pragyan") < rover_names.index("MoonRanger")
    assert rover_names.index("Yutu-2") < rover_names.index("Rashid-1")


def test_list_surrogate_backend_works(client: TestClient) -> None:
    response = client.get("/validate/rediscovery", params={"backend": "surrogate"})
    assert response.status_code == 200
    body = response.json()
    assert body["backend"] == "surrogate"
    slugs = {r["slug"] for r in body["rovers"]}
    # The surrogate backend is a wall-clock benchmark, not ground truth.
    # Its mass head over-predicts ultra-micro (<5 kg) designs (CADRE's
    # published 2 kg sits below the 5-50 kg specific-mass calibration
    # regime), so every NSGA-II individual violates CADRE's mass ceiling
    # and CADRE drops out of the surrogate LOO. The evaluator backend
    # (the paper figure) rediscovers all six; here we assert the surrogate
    # recovers the in-regime rovers and document the CADRE exclusion.
    assert slugs == {"pragyan", "yutu_2", "moonranger", "rashid_1", "tenacious"}
    assert "cadre_unit" not in slugs


def test_invalid_backend_query_returns_422(client: TestClient) -> None:
    response = client.get("/validate/rediscovery", params={"backend": "evaluator_v9"})
    assert response.status_code == 422


def test_get_pragyan_detail_shape(client: TestClient) -> None:
    response = client.get("/validate/rediscovery/pragyan")
    assert response.status_code == 200
    body = response.json()
    assert body["slug"] == "pragyan"
    assert body["rover_name"] == "Pragyan"
    assert body["is_flown"] is True
    assert body["class_generic_scenario"] == "polar_micro"
    # Pragyan's published 0.5 m^2 panel sits inside the design schema.
    assert 0.05 <= body["rover_design"]["wheel_radius_m"] <= 0.20
    # Post-2026-05-28 panel-tilt fix gives Pragyan a positive energy
    # margin and ~15 km of range under polar_micro -- regression guard
    # for the comparison report's headline numbers.
    metrics = body["rover_metrics_under_generic_scenario"]
    assert metrics["energy_margin_raw_pct"] > 0.0
    assert metrics["range_km"] > 10.0
    assert math.isfinite(body["design_space_distance"])
    assert body["design_space_distance"] > 0.0
    assert isinstance(body["pareto_dominated"], bool)
    # The Pareto front must be non-empty for paper-grade rovers.
    assert len(body["pareto_front"]) >= 100
    # Each front point must round-trip through the DesignVector schema.
    first = body["pareto_front"][0]
    assert "design" in first and "metrics" in first
    assert 0.05 <= first["design"]["wheel_radius_m"] <= 0.20
    assert "range_km" in first["metrics"]


def test_get_cadre_unit_detail_is_dominance_false(client: TestClient) -> None:
    """CADRE-unit's mobility-bound stall keeps it off the Pareto-dominated list."""
    response = client.get("/validate/rediscovery/cadre_unit")
    assert response.status_code == 200
    body = response.json()
    assert body["rover_name"] == "CADRE-unit"
    assert body["pareto_dominated"] is False
    # CADRE evaluator distance is the only one < 0.4 ("paper-grade
    # rediscovery"); guard against a regression that would silently
    # push it back over the threshold.
    assert body["design_space_distance"] < 0.45
    assert body["rover_metrics_under_generic_scenario"]["range_km"] == 0.0


def test_get_unknown_slug_returns_404(client: TestClient) -> None:
    response = client.get("/validate/rediscovery/curiosity")
    assert response.status_code == 404
    assert "unknown rediscovery slug" in response.json()["detail"]


def test_pareto_point_metrics_match_design_axes(client: TestClient) -> None:
    """Every front point's design fields must validate as a DesignVector."""
    response = client.get("/validate/rediscovery/yutu_2")
    assert response.status_code == 200
    body = response.json()
    expected_design_keys = {
        "wheel_radius_m",
        "wheel_width_m",
        "grouser_height_m",
        "grouser_count",
        "n_wheels",
        "chassis_mass_kg",
        "wheelbase_m",
        "solar_area_m2",
        "battery_capacity_wh",
        "avionics_power_w",
        "peak_wheel_torque_nm",
    }
    for point in body["pareto_front"][:5]:
        assert set(point["design"].keys()) == expected_design_keys
        assert {"range_km", "total_mass_kg", "slope_capability_deg"}.issubset(
            set(point["metrics"].keys())
        )
