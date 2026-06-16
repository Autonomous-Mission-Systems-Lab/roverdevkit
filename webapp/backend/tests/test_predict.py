"""Smoke tests for ``POST /predict``.

These tests require the quantile-calibration quantile bundle on disk; if it is
missing they skip rather than fail so a contributor without the
artifact can still run the rest of the suite.
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


@pytest.fixture(autouse=True)
def _skip_if_no_v7_1_artifact(surrogate_v7_1_compatible: bool) -> None:
    if not surrogate_v7_1_compatible:
        pytest.skip(
            "schema-v7_1 quantile_bundles.joblib not on disk; skipping "
            "predict tests (pre-v7_1 bundles lack "
            "scenario_operational_duty_cycle and KeyError on the v7_1 "
            "feature row produced by build_feature_row)."
        )


def test_predict_returns_monotone_quantiles(
    client: TestClient,
    sample_design: dict[str, float | int],
) -> None:
    payload = {"design": sample_design, "scenario_name": "equatorial_mare_traverse"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["scenario_name"] == "equatorial_mare_traverse"
    targets = {p["target"] for p in body["predictions"]}
    assert targets == PRIMARY_TARGETS

    for pred in body["predictions"]:
        # repair_crossings defaults to True -> must be monotone.
        assert pred["q05"] <= pred["q50"] <= pred["q95"], pred


def test_predict_feature_row_includes_categoricals(
    client: TestClient,
    sample_design: dict[str, float | int],
) -> None:
    payload = {"design": sample_design, "scenario_name": "polar_prospecting"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    cols = body["feature_row"]["columns"]
    # 27 columns: 11 design (v7 dropped designed_duty_cycle) + 12 scenario
    # numerics (v7_1 added scenario_operational_duty_cycle; v9 added
    # scenario_payload_mass_kg + scenario_payload_power_w) + 4 scenario
    # categoricals.
    assert len(cols) == 27
    assert "scenario_operational_duty_cycle" in cols
    assert "scenario_payload_mass_kg" in cols
    assert "scenario_payload_power_w" in cols
    # Family is forwarded from the scenario name on the canonical four.
    fam_idx = cols.index("scenario_family")
    assert body["feature_row"]["values"][fam_idx] == "polar_prospecting"


def test_predict_payload_override_reaches_feature_row(
    client: TestClient,
    sample_design: dict[str, float | int],
) -> None:
    """Schema v9: a per-call payload override must land in the echoed
    feature row so the surrogate scores the mission's own payload, not
    the scenario default.
    """
    payload = {
        "design": sample_design,
        "scenario_name": "equatorial_mare_traverse",
        "payload_mass_kg": 12.5,
        "payload_power_w": 7.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, response.text
    row = response.json()["feature_row"]
    cols = row["columns"]
    mass_idx = cols.index("scenario_payload_mass_kg")
    power_idx = cols.index("scenario_payload_power_w")
    assert row["values"][mass_idx] == pytest.approx(12.5)
    assert row["values"][power_idx] == pytest.approx(7.0)


def test_predict_mission_duration_override_reaches_feature_row(
    client: TestClient,
    sample_design: dict[str, float | int],
) -> None:
    """A per-call duration override must land in the echoed feature row."""
    payload = {
        "design": sample_design,
        "scenario_name": "equatorial_mare_traverse",
        "mission_duration_earth_days": 21.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, response.text
    row = response.json()["feature_row"]
    duration_idx = row["columns"].index("scenario_mission_duration_earth_days")
    assert row["values"][duration_idx] == pytest.approx(21.0)


def test_predict_rejects_out_of_bounds_payload(
    client: TestClient,
    sample_design: dict[str, float | int],
) -> None:
    response = client.post(
        "/predict",
        json={
            "design": sample_design,
            "scenario_name": "equatorial_mare_traverse",
            "payload_power_w": -1.0,
        },
    )
    assert response.status_code == 422


def test_predict_unknown_scenario_returns_404(
    client: TestClient,
    sample_design: dict[str, float | int],
) -> None:
    payload = {"design": sample_design, "scenario_name": "nope"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 404


def test_predict_rejects_out_of_bounds_design(
    client: TestClient,
    sample_design: dict[str, float | int],
) -> None:
    bad = dict(sample_design)
    bad["wheel_radius_m"] = 5.0  # schema ceiling is 0.20 m
    response = client.post(
        "/predict",
        json={"design": bad, "scenario_name": "equatorial_mare_traverse"},
    )
    # Pydantic v2 returns 422 for body validation failures by default.
    assert response.status_code == 422


def test_predict_raw_quantiles_may_be_non_monotone(
    client: TestClient,
    sample_design: dict[str, float | int],
) -> None:
    """With ``repair_crossings=False`` the API exposes raw model output.

    The contract here is *not* that crossings will appear (they
    usually don't on a single point) but that the repair flag is
    plumbed end to end -- so we just check the response is well-formed.
    """
    payload = {
        "design": sample_design,
        "scenario_name": "highland_slope_capability",
        "repair_crossings": False,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    for pred in body["predictions"]:
        for key in ("q05", "q50", "q95"):
            assert isinstance(pred[key], (int, float))
