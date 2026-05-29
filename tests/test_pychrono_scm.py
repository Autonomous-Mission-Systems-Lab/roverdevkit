"""PyChrono integration PyChrono SCM single-wheel driver tests.

Purpose: validate that ``pychrono_scm.single_wheel_forces_scm`` is
*wired correctly* — i.e., that the rig produces forces that satisfy
basic physics (Fz balance, monotonic DP with slip, correct sign
conventions) and that the numbers land in the same order of
magnitude as the analytical Bekker-Wong model.

This is **not** a validation-against-truth test — SCM and analytical
Bekker-Wong are both models, not ground truth, and the gap between
them is exactly what the real-rover validation correction layer is meant to learn.

The tests are marked ``chrono`` (requires PyChrono) and ``slow``
(simulation wall-clock is seconds, not milliseconds), so they are
skipped by default in the fast ``pytest`` loop and run explicitly
via ``pytest -m 'chrono and slow'`` during PyChrono integration smoke review
and any subsequent SCM-related refactor.
"""

from __future__ import annotations

import pytest

from roverdevkit.terramechanics.bekker_wong import (
    SoilParameters,
    WheelGeometry,
    single_wheel_forces,
)
from roverdevkit.terramechanics.pychrono_scm import (
    ScmConfig,
    is_available,
    single_wheel_forces_scm,
)

pytestmark = [
    pytest.mark.chrono,
    pytest.mark.slow,
    pytest.mark.skipif(not is_available(), reason="PyChrono not installed"),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def apollo_nominal() -> SoilParameters:
    return SoilParameters(
        n=1.0,
        k_c=1.4,
        k_phi=820.0,
        cohesion_kpa=0.17,
        friction_angle_deg=46.0,
    )


@pytest.fixture
def rashid_wheel() -> WheelGeometry:
    return WheelGeometry(radius_m=0.1, width_m=0.06)


@pytest.fixture
def fast_config() -> ScmConfig:
    """Short-runtime config suitable for CI (~0.1 s/call)."""
    return ScmConfig(
        time_step_s=1e-3,
        settle_time_s=0.2,
        drive_time_s=0.8,
        terrain_mesh_res_m=0.02,
        average_window_skip=0.5,
    )


# ---------------------------------------------------------------------------
# Physics-wiring tests
# ---------------------------------------------------------------------------


def test_vertical_force_balance_closes(
    rashid_wheel: WheelGeometry,
    apollo_nominal: SoilParameters,
    fast_config: ScmConfig,
) -> None:
    """Time-averaged vertical soil force must equal the applied load.

    This is the single most important wiring-correctness check — if
    Fz ≠ W_applied, gravity or joint constraints are misconfigured
    and any DP/T numbers are garbage.
    """
    load = 30.0
    telemetry: dict[str, float] = {}
    single_wheel_forces_scm(
        rashid_wheel,
        apollo_nominal,
        vertical_load_n=load,
        slip=0.2,
        config=fast_config,
        telemetry=telemetry,
    )
    assert abs(telemetry["fz_residual_n"]) < 0.05 * load, (
        f"Fz = {telemetry['fz_mean_n']:.2f} N but applied load = {load} N "
        f"(residual {telemetry['fz_residual_n']:+.2f} N)"
    )


def test_drawbar_pull_monotonic_in_driving_slip(
    rashid_wheel: WheelGeometry,
    apollo_nominal: SoilParameters,
    fast_config: ScmConfig,
) -> None:
    """DP must grow monotonically from zero slip through the rising regime."""
    load = 30.0
    slips = [0.0, 0.1, 0.2, 0.35]
    dps = [
        single_wheel_forces_scm(
            rashid_wheel,
            apollo_nominal,
            load,
            s,
            config=fast_config,
        ).drawbar_pull_n
        for s in slips
    ]
    for a, b in zip(dps[:-1], dps[1:], strict=True):
        assert b >= a - 0.2, f"DP should grow with slip, got {dps}"
    # Large-gain check: DP at s=0.35 should clearly exceed DP at s=0.
    assert dps[-1] > dps[0] + 3.0


def test_driving_torque_positive_and_sinkage_sensible(
    rashid_wheel: WheelGeometry,
    apollo_nominal: SoilParameters,
    fast_config: ScmConfig,
) -> None:
    f = single_wheel_forces_scm(
        rashid_wheel,
        apollo_nominal,
        vertical_load_n=30.0,
        slip=0.2,
        config=fast_config,
    )
    assert f.driving_torque_nm > 0.0
    # Sinkage must be strictly positive (wheel sits in the soil) but
    # much smaller than the wheel radius.
    assert 1e-3 < f.sinkage_m < 0.5 * rashid_wheel.radius_m


def test_drawbar_pull_below_mohr_coulomb_limit(
    rashid_wheel: WheelGeometry,
    apollo_nominal: SoilParameters,
    fast_config: ScmConfig,
) -> None:
    """Soil DP can't exceed the Mohr-Coulomb friction envelope μ·W.

    Hitting (or exceeding) this limit is the classic symptom of a
    skidding or otherwise mis-wired wheel — we saw it when the
    rotation-motor sign was backwards.
    """
    import math

    load = 30.0
    f = single_wheel_forces_scm(
        rashid_wheel,
        apollo_nominal,
        load,
        slip=0.3,
        config=fast_config,
    )
    mu_max = math.tan(math.radians(apollo_nominal.friction_angle_deg))
    assert abs(f.drawbar_pull_n) < 0.9 * mu_max * load, (
        f"DP = {f.drawbar_pull_n:.2f} N is saturated at the friction limit "
        f"({mu_max * load:.2f} N); likely a skidding wheel."
    )


# ---------------------------------------------------------------------------
# Cross-check against analytical Bekker-Wong
# ---------------------------------------------------------------------------


def test_scm_and_analytical_agree_in_order_of_magnitude(
    rashid_wheel: WheelGeometry,
    apollo_nominal: SoilParameters,
    fast_config: ScmConfig,
) -> None:
    """Cross-check: SCM vs analytical Bekker-Wong at one operating point.

    The two models will disagree quantitatively — that's expected and
    is precisely the systematic delta that the Path-2 correction layer
    (real-rover validation) will learn to predict. Here we only assert:

      * same sign of DP (both produce traction at driving slip),
      * DP within a factor of 5 (rules out wiring errors of several
        orders of magnitude),
      * driving torque within a factor of 3,
      * sinkage within a factor of 3.

    Bounds are deliberately loose; the real calibration happens in real-rover validation.
    """
    load = 30.0
    slip = 0.2

    f_ana = single_wheel_forces(rashid_wheel, apollo_nominal, load, slip)
    f_scm = single_wheel_forces_scm(
        rashid_wheel,
        apollo_nominal,
        load,
        slip,
        config=fast_config,
    )

    assert (f_ana.drawbar_pull_n > 0) == (f_scm.drawbar_pull_n > 0), (
        f"DP sign mismatch: analytical={f_ana.drawbar_pull_n:+.2f}, SCM={f_scm.drawbar_pull_n:+.2f}"
    )
    assert 0.2 < f_scm.drawbar_pull_n / f_ana.drawbar_pull_n < 5.0, (
        f"DP ratio SCM/analytical = "
        f"{f_scm.drawbar_pull_n / f_ana.drawbar_pull_n:.2f} "
        f"(SCM={f_scm.drawbar_pull_n:.2f}, analytical={f_ana.drawbar_pull_n:.2f})"
    )
    assert 0.3 < f_scm.driving_torque_nm / f_ana.driving_torque_nm < 3.0, (
        f"T ratio SCM/analytical = {f_scm.driving_torque_nm / f_ana.driving_torque_nm:.2f}"
    )
    assert 0.3 < f_scm.sinkage_m / f_ana.sinkage_m < 3.0, (
        f"sinkage ratio SCM/analytical = {f_scm.sinkage_m / f_ana.sinkage_m:.2f}"
    )


# ---------------------------------------------------------------------------
# Go/no-go performance benchmark
# ---------------------------------------------------------------------------


def test_scm_wall_clock_under_path2_budget(
    rashid_wheel: WheelGeometry,
    apollo_nominal: SoilParameters,
    fast_config: ScmConfig,
) -> None:
    """Formal PyChrono integration smoke gate: wall-clock per simulated-wheel-second
    must stay under the 10 s / wheel-second runtime budget.

    Measured on an Apple M-series laptop at the default (δ=15 mm) mesh,
    this is typically ~0.08 s/wheel-s — two orders of magnitude of headroom.
    Use a generous 2 s/wheel-s ceiling here to leave room for CI runners
    that may be slower than dev machines.
    """
    telemetry: dict[str, float] = {}
    single_wheel_forces_scm(
        rashid_wheel,
        apollo_nominal,
        30.0,
        0.2,
        config=fast_config,
        telemetry=telemetry,
    )
    sim_time_s = fast_config.settle_time_s + fast_config.drive_time_s
    ratio = telemetry["wall_clock_s"] / sim_time_s
    assert ratio < 2.0, (
        f"wall-clock per simulated-wheel-second = {ratio:.2f} "
        f"exceeds CI budget of 2.0; Path-2 may need re-evaluation"
    )
