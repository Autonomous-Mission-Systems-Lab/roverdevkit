"""Smoke tests for ``POST /evaluate``.

These run the analytical mission evaluator end-to-end. Unlike the
predict tests they do *not* depend on the quantile-calibration artifact.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

PRIMARY_TARGETS = {
    "range_km",
    "energy_margin_raw_pct",
    "slope_capability_deg",
    "total_mass_kg",
}


def test_evaluate_returns_all_primary_targets(
    client: TestClient,
    sample_design: dict[str, float | int],
) -> None:
    payload = {"design": sample_design, "scenario_name": "equatorial_mare_traverse"}
    response = client.post("/evaluate", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["scenario_name"] == "equatorial_mare_traverse"
    targets = {m["target"] for m in body["metrics"]}
    assert targets == PRIMARY_TARGETS
    for metric in body["metrics"]:
        assert isinstance(metric["value"], (int, float))

    thermal = body["thermal"]
    for key in (
        "survives",
        "peak_sun_temp_c",
        "lunar_night_temp_c",
        "min_operating_temp_c",
        "max_operating_temp_c",
        "rhu_power_w",
        "hibernation_power_w",
        "surface_area_m2",
        "hot_case_ok",
        "cold_case_ok",
    ):
        assert key in thermal
    # The default architecture has a -30/+50 °C envelope and these
    # are the limits the survival flag is judged against.
    assert thermal["min_operating_temp_c"] == -30.0
    assert thermal["max_operating_temp_c"] == 50.0

    # Schema v6 (v6 schema update): the per-evaluation drivetrain diagnostic
    # was renamed from ``motor_torque`` to ``stall`` and exposes the
    # explicit slip / capacity headroom rather than the v5 OK/NOT-OK
    # composite. See ``StallDiagnosticOut`` in webapp.backend.schemas.
    stall = body["stall"]
    for key in (
        "stalled",
        "peak_torque_demand_nm",
        "peak_torque_capacity_nm",
    ):
        assert key in stall
    assert stall["peak_torque_demand_nm"] >= 0.0
    assert stall["peak_torque_capacity_nm"] > 0.0

    # Schema v6 also surfaces the runtime-derived effective duty cycle
    # and cruise speed at the top level so the frontend can show what
    # the evaluator actually used (vs. the design's δ_des).
    assert "effective_duty_cycle" in body
    assert 0.0 <= body["effective_duty_cycle"] <= 0.6
    assert "cruise_speed_mps" in body
    assert body["cruise_speed_mps"] >= 0.0

    assert body["elapsed_ms"] > 0


def test_evaluate_thermal_cold_case_drives_failure_for_no_rhu_design(
    client: TestClient,
    sample_design: dict[str, float | int],
) -> None:
    """The default architecture has 0 W RHU; cold case should be the failing one.

    With no RHU and 2 W of hibernation power, a 0.2-ish m² enclosure
    radiates to ~133 K (well below the −30 °C limit) and the hot case
    sits comfortably under +50 °C at any latitude. The dialog leans on
    this distinction to explain *why* survival fails, so we pin it
    here.
    """
    payload = {"design": sample_design, "scenario_name": "equatorial_mare_traverse"}
    response = client.post("/evaluate", json=payload)
    assert response.status_code == 200
    thermal = response.json()["thermal"]
    if not thermal["survives"]:
        assert not thermal["cold_case_ok"]
    # Hot case should never be the failure for this sample design at
    # equatorial latitude (sanity guard against a regression that
    # silently flips the model).
    assert thermal["hot_case_ok"]


def test_evaluate_payload_override_increases_total_mass(
    client: TestClient,
    sample_design: dict[str, float | int],
) -> None:
    """Schema v9: the ``payload_mass_kg`` override is a top-level mass line
    item, so a non-zero override raises ``total_mass_kg`` ~one-for-one and
    leaves the other primary targets at or below their no-payload values.
    """
    base = {"design": sample_design, "scenario_name": "equatorial_mare_traverse"}
    base_resp = client.post("/evaluate", json={**base, "payload_mass_kg": 0.0})
    heavy_resp = client.post("/evaluate", json={**base, "payload_mass_kg": 10.0})
    assert base_resp.status_code == 200, base_resp.text
    assert heavy_resp.status_code == 200, heavy_resp.text

    base_mass = {m["target"]: m["value"] for m in base_resp.json()["metrics"]}[
        "total_mass_kg"
    ]
    heavy_mass = {m["target"]: m["value"] for m in heavy_resp.json()["metrics"]}[
        "total_mass_kg"
    ]
    # Payload sits outside the dry-mass growth margin, so the delta is the
    # payload itself (no extra margin applied on top).
    assert heavy_mass == pytest.approx(base_mass + 10.0, abs=1e-6)


def test_evaluate_payload_power_override_reduces_range(
    client: TestClient,
    sample_design: dict[str, float | int],
) -> None:
    """Schema v9: ``payload_power_w`` adds to the continuous ops-time load,
    so a non-zero override never increases range and typically shrinks it.
    """
    base = {"design": sample_design, "scenario_name": "equatorial_mare_traverse"}
    quiet = client.post("/evaluate", json={**base, "payload_power_w": 0.0})
    noisy = client.post("/evaluate", json={**base, "payload_power_w": 25.0})
    assert quiet.status_code == 200, quiet.text
    assert noisy.status_code == 200, noisy.text
    quiet_range = {m["target"]: m["value"] for m in quiet.json()["metrics"]}["range_km"]
    noisy_range = {m["target"]: m["value"] for m in noisy.json()["metrics"]}["range_km"]
    assert noisy_range <= quiet_range + 1e-9


def test_evaluate_rejects_out_of_bounds_payload(
    client: TestClient,
    sample_design: dict[str, float | int],
) -> None:
    """Payload overrides are bounded ``[0, 30]`` at the HTTP boundary."""
    response = client.post(
        "/evaluate",
        json={
            "design": sample_design,
            "scenario_name": "equatorial_mare_traverse",
            "payload_mass_kg": 999.0,
        },
    )
    assert response.status_code == 422


def test_evaluate_unknown_scenario_returns_404(
    client: TestClient,
    sample_design: dict[str, float | int],
) -> None:
    payload = {"design": sample_design, "scenario_name": "no_such_scenario"}
    response = client.post("/evaluate", json=payload)
    assert response.status_code == 404


def test_evaluate_rejects_out_of_bounds_design(
    client: TestClient,
    sample_design: dict[str, float | int],
) -> None:
    bad = dict(sample_design)
    bad["wheel_radius_m"] = 5.0
    response = client.post(
        "/evaluate",
        json={"design": bad, "scenario_name": "equatorial_mare_traverse"},
    )
    assert response.status_code == 422


def test_evaluate_values_match_primary_metrics_shape(
    client: TestClient,
    sample_design: dict[str, float | int],
) -> None:
    """Sanity-check the projection of ``MissionMetrics`` onto the four primary targets.

    Range and total mass are strictly positive for every well-formed
    scenario; slope is bounded above by 90°; energy margin is unbounded
    but should be finite. This is a coarse "no NaN snuck through" guard.
    """
    payload = {"design": sample_design, "scenario_name": "polar_prospecting"}
    response = client.post("/evaluate", json=payload)
    assert response.status_code == 200
    body = response.json()
    by_target = {m["target"]: m["value"] for m in body["metrics"]}

    assert by_target["total_mass_kg"] > 0
    assert by_target["range_km"] >= 0
    assert 0 <= by_target["slope_capability_deg"] <= 90
    assert by_target["energy_margin_raw_pct"] == by_target["energy_margin_raw_pct"]  # not NaN


def test_evaluate_and_predict_agree_within_surrogate_noise_floor(
    client: TestClient,
    sample_design: dict[str, float | int],
    surrogate_v7_1_compatible: bool,
) -> None:
    """The surrogate's median should track the evaluator within R²-noise.

    On the canonical equatorial-mare scenario for the Yutu-2-ish
    sample design, the tuned-median tuned median has R² ≥ 0.99 on every
    primary target. We pick a generous tolerance per target rather
    than assert exact equality so this test does not flake on
    XGBoost-version churn or harmless quantile-head retrains.
    """
    if not surrogate_v7_1_compatible:
        pytest.skip(
            "schema-v7_1 quantile_bundles.joblib not on disk; pre-v7_1 "
            "bundles lack scenario_operational_duty_cycle and KeyError "
            "on the v7_1 feature row."
        )
    payload = {"design": sample_design, "scenario_name": "equatorial_mare_traverse"}
    eval_resp = client.post("/evaluate", json=payload)
    pred_resp = client.post("/predict", json=payload)
    assert eval_resp.status_code == 200
    if pred_resp.status_code == 503:
        # Quantile bundles missing (mirrors the predict-test skip path).
        return
    assert pred_resp.status_code == 200

    evaluator = {m["target"]: m["value"] for m in eval_resp.json()["metrics"]}
    surrogate = {p["target"]: p["q50"] for p in pred_resp.json()["predictions"]}

    # Per-target relative tolerance on the median. Energy margin runs
    # large positive on equatorial-mare so we use absolute tolerance
    # (a 5 pp gap on a 600 % margin is still <1 % relative error).
    # The slope tolerance is set to ~2x the v9 surrogate's overall test
    # RMSE (0.930 deg) divided by a typical equatorial-mare sample-design
    # slope_capability (~22 deg) — i.e. tight enough to catch wiring bugs
    # but loose enough not to flake on a single-point tail residual at
    # the surrogate noise floor. Widened from 0.08 (v6) to 0.10 (v9)
    # because the v9 median head is marginally noisier on slope after the
    # payload-feature retrain (test R² 0.978). See
    # ``reports/surrogate_v9/median_sanity.csv``.
    # total_mass is a near-analytic function of design + payload, so the
    # median head learns it to high precision (test R² 0.999, RMSE
    # 0.567 kg). The 0.04 rel tol (~1.8 kg at this ~44 kg sample design)
    # is ~3x RMSE — a single-point tail allowance now that payload is an
    # extra LHS input adding a little variance, still tight enough to
    # catch a units / wiring regression.
    rel_tol = {
        "range_km": 0.10,
        "slope_capability_deg": 0.10,
        "total_mass_kg": 0.04,
    }
    for tgt, tol in rel_tol.items():
        e = evaluator[tgt]
        s = surrogate[tgt]
        assert abs(e - s) <= max(tol * abs(e), 1e-3), (tgt, e, s)
    # Energy margin: tolerate a 50-pp gap in absolute terms.
    assert abs(evaluator["energy_margin_raw_pct"] - surrogate["energy_margin_raw_pct"]) <= 50
