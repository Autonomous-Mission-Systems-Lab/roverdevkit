"""Smoke tests for ``POST /sweep``.

The evaluator backend has no artifact dependency beyond the SCM
correction (already optional); the surrogate backend needs the W8
step-4 quantile bundle. Tests that require the bundle skip when it
is missing so a contributor without the artifact can still run the
evaluator path locally.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def _payload(
    design: dict[str, float | int],
    *,
    target: str = "range_km",
    backend: str = "evaluator",
    x_n: int = 4,
    y_axis: dict[str, float | int] | None = None,
    scenario: str = "equatorial_mare_traverse",
) -> dict[str, object]:
    body: dict[str, object] = {
        "target": target,
        "x_axis": {
            "variable": "wheel_radius_m",
            "lo": 0.08,
            "hi": 0.18,
            "n_points": x_n,
        },
        "base_design": design,
        "scenario_name": scenario,
        "backend": backend,
    }
    if y_axis is not None:
        body["y_axis"] = y_axis
    return body


def test_sweep_evaluator_1d_returns_one_value_per_grid_point(
    client: TestClient,
    sample_design: dict[str, float | int],
) -> None:
    response = client.post(
        "/sweep", json=_payload(sample_design, x_n=4, backend="evaluator")
    )
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["target"] == "range_km"
    assert body["x_variable"] == "wheel_radius_m"
    assert body["y_variable"] is None
    assert body["y_values"] is None
    assert len(body["x_values"]) == 4
    assert len(body["z_values"]) == 4
    assert all(isinstance(v, (int, float)) for v in body["z_values"])
    assert body["backend_used"] == "evaluator"
    assert body["backend_requested"] == "evaluator"
    assert body["n_cells"] == 4
    assert body["elapsed_ms"] >= 0.0


def test_sweep_evaluator_2d_returns_y_outer_x_inner_matrix(
    client: TestClient,
    sample_design: dict[str, float | int],
) -> None:
    response = client.post(
        "/sweep",
        json=_payload(
            sample_design,
            x_n=3,
            backend="evaluator",
            y_axis={
                "variable": "solar_area_m2",
                "lo": 0.4,
                "hi": 0.8,
                "n_points": 2,
            },
        ),
    )
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["y_variable"] == "solar_area_m2"
    assert len(body["y_values"]) == 2
    assert len(body["x_values"]) == 3
    z = body["z_values"]
    assert len(z) == 2  # outer = y
    assert all(len(row) == 3 for row in z)  # inner = x
    assert body["n_cells"] == 6
    assert body["backend_used"] == "evaluator"


def test_sweep_surrogate_backend_when_artifact_present(
    client: TestClient,
    sample_design: dict[str, float | int],
    surrogate_v7_1_compatible: bool,
) -> None:
    if not surrogate_v7_1_compatible:
        pytest.skip(
            "schema-v7_1 quantile_bundles.joblib not on disk; skipping "
            "surrogate sweep."
        )
    response = client.post(
        "/sweep", json=_payload(sample_design, x_n=8, backend="surrogate")
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["backend_used"] == "surrogate"
    assert body["used_scm_correction"] is False  # surrogate path
    assert len(body["z_values"]) == 8


def test_sweep_rejects_unknown_axis_variable(
    client: TestClient,
    sample_design: dict[str, float | int],
) -> None:
    payload = _payload(sample_design)
    payload["x_axis"]["variable"] = "n_wheels"  # excluded from sweepables
    response = client.post("/sweep", json=payload)
    # The Pydantic-level check passes the string through, but the
    # roverdevkit-side guard fires either at request validation
    # (route) or at SweepSpec.__post_init__; either way we get a 422.
    assert response.status_code == 422, response.text


def test_sweep_evaluator_hard_limit_returns_422(
    client: TestClient,
    sample_design: dict[str, float | int],
) -> None:
    # 200 * 200 = 40k cells on the evaluator path -> trips hard limit.
    payload = _payload(
        sample_design,
        x_n=200,
        backend="evaluator",
        y_axis={
            "variable": "solar_area_m2",
            "lo": 0.4,
            "hi": 0.8,
            "n_points": 200,
        },
    )
    response = client.post("/sweep", json=payload)
    assert response.status_code == 422, response.text
    assert "evaluator hard limit" in response.json()["detail"]


def test_sweep_unknown_scenario_returns_404(
    client: TestClient,
    sample_design: dict[str, float | int],
) -> None:
    response = client.post(
        "/sweep",
        json=_payload(sample_design, scenario="not_a_real_scenario"),
    )
    assert response.status_code == 404, response.text


def test_sweep_honours_operational_duty_cycle_override(
    client: TestClient,
    sample_design: dict[str, float | int],
) -> None:
    """SCHEMA_VERSION v7_1: the δ_ops override flows through to the sweep route.

    Equatorial-mare with continuous sun is duty-bound on
    ``range_km``: at the family default (δ_ops = 0.30) the rover
    runs hard; cutting δ_ops to 0.05 should slash range
    proportionally on every cell (range_km scales linearly with
    δ_eff until it hits the non-binding traverse-distance cap or
    the energy throttle).
    """
    base = _payload(sample_design, x_n=4, backend="evaluator", scenario="equatorial_mare_traverse")
    overridden_low = dict(base)
    overridden_low["operational_duty_cycle"] = 0.05
    overridden_high = dict(base)
    overridden_high["operational_duty_cycle"] = 0.30

    low_resp = client.post("/sweep", json=overridden_low)
    high_resp = client.post("/sweep", json=overridden_high)
    assert low_resp.status_code == 200, low_resp.text
    assert high_resp.status_code == 200, high_resp.text

    low_z = low_resp.json()["z_values"]
    high_z = high_resp.json()["z_values"]
    # Identical designs / scenarios except for δ_ops; the higher
    # duty cycle must move at least one cell, otherwise the
    # override silently dropped on the floor.
    assert low_z != high_z
    # Sanity-check: the higher-δ_ops grid has every cell ≥ the
    # lower-δ_ops grid (range monotone in δ_eff).
    assert all(h >= ll for ll, h in zip(low_z, high_z, strict=True))


def test_sweep_rejects_out_of_bounds_operational_duty_cycle(
    client: TestClient,
    sample_design: dict[str, float | int],
) -> None:
    """SchemaField bounds [0, 0.6] on operational_duty_cycle reject 0.9 → 422."""
    payload = _payload(sample_design)
    payload["operational_duty_cycle"] = 0.9
    response = client.post("/sweep", json=payload)
    assert response.status_code == 422, response.text
