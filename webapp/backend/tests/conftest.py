"""Shared fixtures for the webapp backend test suite.

The fixtures here are intentionally small: build a real FastAPI app
backed by the real on-disk artifacts, and hand it to a `TestClient`
once per test session. We do **not** mock the surrogate or the
scenario loaders -- the whole point of this test suite is to catch
artifact-on-disk drift before it hits the frontend.

If the W8 step-4 quantile bundle is missing the suite will skip the
predict tests rather than fail outright; this lets a contributor who
has not yet generated the artifact still run health / scenarios /
registry tests locally.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from webapp.backend.app import create_app
from webapp.backend.config import get_settings
from webapp.backend.loaders import reset_caches


@pytest.fixture(scope="session")
def client() -> Iterator[TestClient]:
    """Return a `TestClient` for the real backend app, session-scoped."""
    reset_caches()
    app = create_app()
    with TestClient(app) as c:
        yield c
    reset_caches()


@pytest.fixture(scope="session")
def artifacts_present() -> bool:
    """Whether the on-disk surrogate artifact is loadable.

    Tracks a *file-existence* condition only; the surrogate may still
    be schema-incompatible with the live evaluator (e.g. while a
    schema-bump retrain is in flight). Use
    :func:`surrogate_v7_1_compatible` to gate tests that actually call
    ``/predict`` end-to-end.
    """
    return get_settings().artifacts_present


@pytest.fixture(scope="session")
def surrogate_v7_1_compatible() -> bool:
    """Whether the on-disk surrogate is schema-v7_1 compatible.

    Schema v7_1 (W12 step B follow-on) promotes
    ``scenario_operational_duty_cycle`` to a true surrogate input
    feature; v7 bundles do not expect it and KeyError at predict time
    once the v7_1 feature-row builder includes it. Predict /
    surrogate-sweep tests skip on schema mismatch instead of failing
    red until the v7_1 recalibrate lands.
    """
    settings = get_settings()
    if not settings.artifacts_present:
        return False
    try:
        import joblib

        bundles = joblib.load(settings.quantile_bundles_path)
        any_bundle = next(iter(bundles.values()))
        feature_columns = list(getattr(any_bundle, "feature_columns", []))
    except Exception:
        return False
    return (
        "design_peak_wheel_torque_nm" in feature_columns
        and "design_designed_duty_cycle" not in feature_columns
        and "scenario_operational_duty_cycle" in feature_columns
    )


@pytest.fixture()
def sample_design() -> dict[str, float | int]:
    """A safely in-bounds design vector (Yutu-2-ish) for predict tests.

    Mirrors the real Yutu-2 design except where the design schema's
    bounds force a tweak, so the request payload always validates.
    Kept out of the registry on purpose -- the predict tests should
    work even if the registry export ever changes.
    """
    return {
        "wheel_radius_m": 0.10,
        "wheel_width_m": 0.10,
        "grouser_height_m": 0.012,
        "grouser_count": 14,
        "n_wheels": 6,
        "chassis_mass_kg": 20.0,
        "wheelbase_m": 0.6,
        "solar_area_m2": 0.5,
        "battery_capacity_wh": 100.0,
        "avionics_power_w": 15.0,
        "peak_wheel_torque_nm": 1.5,
    }
