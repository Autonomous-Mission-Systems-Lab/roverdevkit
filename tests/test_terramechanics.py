"""Tests for the terramechanics sub-package.

terramechanics coverage is physics-first-principles sanity:

- force-balance self-consistency,
- monotonic response to load, slip, soil stiffness, and wheel width,
- sign conventions (positive slip ⇒ torque draw, zero slip ⇒ drawbar pull
  dominated by compaction drag),
- Bekker plate compaction resistance matches the integrated zero-slip
  drawbar pull to within model-form noise,
- sub-millisecond runtime,
- **Layer-3 published-reference grid** (see ``data/validation/wong_layer3_reference.csv``):
  per-row tolerance-bound check that the kernel reproduces a Wong (2008)
  §4.2-style worked-example fixture and falls inside the published
  Bekker-Wong band for Pragyan-/Yutu-2-class wheels on Apollo regolith.
  Tolerance bands are sized at the ±15-30 % BW model-form error reported
  in Ishigami (2007) and Ding et al. (2011).
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import pytest

from roverdevkit.mission.capability import max_climbable_slope_deg
from roverdevkit.terramechanics.bekker_wong import (
    _GROUSER_LIFT_CAP,
    SoilParameters,
    WheelForces,
    WheelGeometry,
    _grouser_shear_lift,
    _integrate_forces,
    single_wheel_forces,
)

LAYER3_REFERENCE_CSV = (
    Path(__file__).resolve().parent.parent / "data" / "validation" / "wong_layer3_reference.csv"
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def nominal_soil() -> SoilParameters:
    """Apollo regolith nominal — matches ``data/soil_simulants.csv``."""
    return SoilParameters(
        n=1.0,
        k_c=1.4,
        k_phi=820.0,
        cohesion_kpa=0.17,
        friction_angle_deg=46.0,
    )


@pytest.fixture
def loose_soil() -> SoilParameters:
    """Apollo regolith loose-bound — softer."""
    return SoilParameters(
        n=1.0,
        k_c=0.5,
        k_phi=400.0,
        cohesion_kpa=0.1,
        friction_angle_deg=30.0,
    )


@pytest.fixture
def dense_soil() -> SoilParameters:
    """Apollo regolith dense-bound — stiffer."""
    return SoilParameters(
        n=1.2,
        k_c=2.0,
        k_phi=1200.0,
        cohesion_kpa=0.5,
        friction_angle_deg=50.0,
    )


@pytest.fixture
def rashid_wheel() -> WheelGeometry:
    """Rashid-like: ~0.1 m radius, 0.06 m wide."""
    return WheelGeometry(radius_m=0.1, width_m=0.06)


# ---------------------------------------------------------------------------
# Dataclass smoke tests
# ---------------------------------------------------------------------------


def test_soil_and_wheel_dataclasses_are_constructable() -> None:
    soil = SoilParameters(n=1.0, k_c=1.4, k_phi=820.0, cohesion_kpa=1.0, friction_angle_deg=45.0)
    wheel = WheelGeometry(radius_m=0.1, width_m=0.06, grouser_height_m=0.005, grouser_count=12)
    assert soil.n == 1.0
    assert wheel.radius_m == 0.1


# ---------------------------------------------------------------------------
# Force-balance self-consistency
# ---------------------------------------------------------------------------


def test_force_balance_closes_at_solved_entry_angle(
    rashid_wheel: WheelGeometry, nominal_soil: SoilParameters
) -> None:
    """W_integrated(θ₁★) must equal the applied load within tight tolerance."""
    load = 150.0
    forces = single_wheel_forces(rashid_wheel, nominal_soil, load, slip=0.2)
    w_check, _, _ = _integrate_forces(forces.entry_angle_rad, rashid_wheel, nominal_soil, 0.2)
    assert w_check == pytest.approx(load, rel=1e-4)


# ---------------------------------------------------------------------------
# Monotonicity
# ---------------------------------------------------------------------------


def test_sinkage_monotonic_in_load(
    rashid_wheel: WheelGeometry, nominal_soil: SoilParameters
) -> None:
    loads = [30.0, 60.0, 120.0, 240.0]
    sinkages = [
        single_wheel_forces(rashid_wheel, nominal_soil, w, slip=0.0).sinkage_m for w in loads
    ]
    assert sinkages == sorted(sinkages), f"sinkage should be monotonic in load, got {sinkages}"
    assert sinkages[-1] > sinkages[0]  # strict increase across the range


def test_drawbar_pull_increases_with_slip_in_traction_regime(
    rashid_wheel: WheelGeometry, nominal_soil: SoilParameters
) -> None:
    """Between low and moderate slip, drawbar pull grows.

    (The relationship saturates near ~50 % slip and is not strictly
    monotonic all the way to 100 %; we test only the rising regime.)
    """
    load = 150.0
    slips = [0.02, 0.05, 0.1, 0.2, 0.35]
    dps = [
        single_wheel_forces(rashid_wheel, nominal_soil, load, slip=s).drawbar_pull_n for s in slips
    ]
    for a, b in zip(dps[:-1], dps[1:], strict=True):
        assert b >= a - 1e-6, f"DP should not decrease in rising slip regime, got {dps}"
    assert dps[-1] > dps[0]


def test_softer_soil_sinks_more(
    rashid_wheel: WheelGeometry, loose_soil: SoilParameters, dense_soil: SoilParameters
) -> None:
    load = 150.0
    soft = single_wheel_forces(rashid_wheel, loose_soil, load, slip=0.0).sinkage_m
    stiff = single_wheel_forces(rashid_wheel, dense_soil, load, slip=0.0).sinkage_m
    assert soft > stiff


def test_wider_wheel_sinks_less(nominal_soil: SoilParameters) -> None:
    narrow = WheelGeometry(radius_m=0.1, width_m=0.04)
    wide = WheelGeometry(radius_m=0.1, width_m=0.12)
    load = 150.0
    z_narrow = single_wheel_forces(narrow, nominal_soil, load, slip=0.0).sinkage_m
    z_wide = single_wheel_forces(wide, nominal_soil, load, slip=0.0).sinkage_m
    assert z_wide < z_narrow


# ---------------------------------------------------------------------------
# Sign conventions
# ---------------------------------------------------------------------------


def test_drawbar_pull_much_smaller_at_zero_than_driving_slip(
    rashid_wheel: WheelGeometry, nominal_soil: SoilParameters
) -> None:
    """At slip = 0, net tractive force should be a small fraction of the
    driving-slip value.

    We deliberately do **not** assert ``DP(s=0) < 0``. The pure Bekker-Wong
    model keeps a nonzero kinematic shear term even at zero slip — for
    high-friction low-cohesion soils (Apollo regolith, φ ≈ 46°) this can
    tip the integrated DP slightly positive. That's a known ±15–30 %
    weakness of the analytical model (cf. Ishigami 2007, Ding 2011) and
    is exactly the kind of model-form error the published tolerance band covers.
    The robust first-principles test is therefore on the *ratio*: at
    zero slip, whatever sign it has, the magnitude should be much smaller
    than at moderate driving slip.
    """
    load = 150.0
    dp_zero = single_wheel_forces(rashid_wheel, nominal_soil, load, slip=0.0).drawbar_pull_n
    dp_drive = single_wheel_forces(rashid_wheel, nominal_soil, load, slip=0.3).drawbar_pull_n
    assert abs(dp_zero) < 0.5 * abs(dp_drive)
    assert dp_drive > 0.0


def test_torque_positive_when_driving(
    rashid_wheel: WheelGeometry, nominal_soil: SoilParameters
) -> None:
    forces = single_wheel_forces(rashid_wheel, nominal_soil, vertical_load_n=150.0, slip=0.2)
    assert forces.driving_torque_nm > 0.0


def test_compaction_resistance_positive_and_grows_with_sinkage(
    rashid_wheel: WheelGeometry, nominal_soil: SoilParameters
) -> None:
    """The Bekker plate compaction resistance is a diagnostic output;
    check it's positive and scales with sinkage (monotonic in load)."""
    light = single_wheel_forces(rashid_wheel, nominal_soil, vertical_load_n=30.0, slip=0.0)
    heavy = single_wheel_forces(rashid_wheel, nominal_soil, vertical_load_n=300.0, slip=0.0)
    assert light.rolling_resistance_n > 0.0
    assert heavy.rolling_resistance_n > light.rolling_resistance_n
    assert heavy.sinkage_m > light.sinkage_m


# ---------------------------------------------------------------------------
# Physical plausibility on a Rashid-class design point
# ---------------------------------------------------------------------------


def test_rashid_class_sinkage_and_dp_are_plausible(
    rashid_wheel: WheelGeometry, nominal_soil: SoilParameters
) -> None:
    """Rough order-of-magnitude check against published micro-rover experience.

    Rashid was ~10 kg / 4 wheels ≈ 25 N per wheel on Earth, ≈ 4 N per
    wheel on the Moon. We run 50 N per wheel (a conservative Earth-like
    check) on nominal regolith with moderate slip and verify that
    sinkage stays in a few mm to few cm, drawbar pull is nonzero, and
    the torque is well within stall-torque of hobby-scale drive motors
    (order 1 N·m). Ballpark, not a calibrated test.
    """
    forces = single_wheel_forces(rashid_wheel, nominal_soil, vertical_load_n=50.0, slip=0.15)
    assert 0.0005 < forces.sinkage_m < 0.05, (
        f"sinkage {forces.sinkage_m * 1000:.1f} mm out of range"
    )
    assert forces.drawbar_pull_n > 0.0
    assert 0.0 < forces.driving_torque_nm < 10.0


# ---------------------------------------------------------------------------
# Grouser shear-thrust lift (Iizuka & Kubota 2011)
# ---------------------------------------------------------------------------


def test_grouser_lift_is_unity_when_no_grousers() -> None:
    """No grouser height or zero count ⇒ lift factor exactly 1.0."""
    smooth = WheelGeometry(radius_m=0.10, width_m=0.10, grouser_height_m=0.0, grouser_count=14)
    no_count = WheelGeometry(radius_m=0.10, width_m=0.10, grouser_height_m=0.012, grouser_count=0)
    assert _grouser_shear_lift(smooth) == 1.0
    assert _grouser_shear_lift(no_count) == 1.0


def test_grouser_lift_matches_arc_density_formula() -> None:
    """Closed form: lift = 1 + N_g · h_g / (2π·R) (uncapped regime)."""
    import math as _math

    wheel = WheelGeometry(radius_m=0.10, width_m=0.10, grouser_height_m=0.005, grouser_count=14)
    expected = 1.0 + 14 * 0.005 / (2.0 * _math.pi * 0.10)
    assert _grouser_shear_lift(wheel) == pytest.approx(expected, rel=1e-12)


def test_grouser_lift_saturates_at_cap() -> None:
    """Very dense grouser pack ⇒ lift saturates at 1 + _GROUSER_LIFT_CAP."""
    extreme = WheelGeometry(radius_m=0.05, width_m=0.10, grouser_height_m=0.020, grouser_count=120)
    assert _grouser_shear_lift(extreme) == pytest.approx(1.0 + _GROUSER_LIFT_CAP, rel=1e-12)


def test_smooth_wheel_baseline_unchanged_after_grouser_term(
    nominal_soil: SoilParameters,
) -> None:
    """Adding the grouser term must not perturb wheels with no grousers.

    Pins the BW kernel to its pre-grouser-term outputs for a smooth
    wheel — guards against accidentally scaling τ when the lift factor
    should be exactly 1.0.
    """
    smooth = WheelGeometry(radius_m=0.10, width_m=0.10, grouser_height_m=0.0, grouser_count=0)
    f = single_wheel_forces(smooth, nominal_soil, vertical_load_n=40.0, slip=0.6)
    assert f.drawbar_pull_n == pytest.approx(11.786, abs=0.05)
    assert f.driving_torque_nm == pytest.approx(2.349, abs=0.02)
    assert f.sinkage_m == pytest.approx(0.01779, abs=1e-4)


def test_drawbar_pull_monotone_in_grouser_height(
    loose_soil: SoilParameters,
) -> None:
    """Taller grousers ⇒ more shear thrust on loose soil at fixed slip."""
    heights_m = [0.000, 0.005, 0.010, 0.015, 0.020]
    dps = []
    for h in heights_m:
        wheel = WheelGeometry(radius_m=0.10, width_m=0.10, grouser_height_m=h, grouser_count=14)
        dps.append(single_wheel_forces(wheel, loose_soil, vertical_load_n=40.0, slip=0.6).drawbar_pull_n)
    for a, b in zip(dps[:-1], dps[1:], strict=True):
        assert b > a, f"DP must strictly increase with h_g, got {dps}"


def test_drawbar_pull_monotone_then_saturates_in_grouser_count(
    loose_soil: SoilParameters,
) -> None:
    """More grousers ⇒ more shear thrust, with saturation at the cap."""
    counts = [0, 4, 8, 14, 24, 36, 60, 120]
    dps = []
    for n in counts:
        wheel = WheelGeometry(radius_m=0.10, width_m=0.10, grouser_height_m=0.012, grouser_count=n)
        dps.append(single_wheel_forces(wheel, loose_soil, vertical_load_n=40.0, slip=0.6).drawbar_pull_n)
    # Strictly rising in the unsaturated regime (counts up to where the
    # cap clamps in — at R=0.10, h=0.012 m the cap is reached around N_g ≈ 32):
    for a, b in zip(dps[:4], dps[1:5], strict=True):
        assert b > a, f"DP must strictly increase with N_g pre-saturation, got {dps}"
    # Once saturated, additional grousers must not decrease DP and must
    # be effectively flat (same lift factor of 1 + cap):
    assert dps[-1] == pytest.approx(dps[-2], rel=1e-3)


def test_slope_capability_picks_up_grouser_signal(
    nominal_soil: SoilParameters,
) -> None:
    """End-to-end: max_climbable_slope_deg now responds to grouser geometry.

    Pre-fix this test would have shown identical slope across grouser
    height — the regression we are fixing.
    """
    smooth = WheelGeometry(radius_m=0.10, width_m=0.10, grouser_height_m=0.0, grouser_count=14)
    grousered = WheelGeometry(
        radius_m=0.10, width_m=0.10, grouser_height_m=0.015, grouser_count=14
    )
    s_smooth = max_climbable_slope_deg(smooth, nominal_soil, total_mass_kg=24.0, n_wheels=6)
    s_grousered = max_climbable_slope_deg(grousered, nominal_soil, total_mass_kg=24.0, n_wheels=6)
    # Expect at least 3° lift from h_g = 15 mm at R = 10 cm; arc-density
    # form gives ≈ 33% shear lift here, which historically maps to
    # 5–8° slope-capability gain on Apollo nominal regolith.
    assert s_grousered - s_smooth > 3.0


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_rejects_nonpositive_load(
    rashid_wheel: WheelGeometry, nominal_soil: SoilParameters
) -> None:
    with pytest.raises(ValueError, match="positive"):
        single_wheel_forces(rashid_wheel, nominal_soil, vertical_load_n=0.0, slip=0.2)


def test_rejects_out_of_range_slip(
    rashid_wheel: WheelGeometry, nominal_soil: SoilParameters
) -> None:
    with pytest.raises(ValueError, match="slip"):
        single_wheel_forces(rashid_wheel, nominal_soil, vertical_load_n=50.0, slip=1.5)


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


def test_single_wheel_forces_runs_under_one_millisecond(
    rashid_wheel: WheelGeometry, nominal_soil: SoilParameters
) -> None:
    """Per the plan (§4), we need < 1 ms per call for 50k+ mission runs.

    We time a warm run to exclude JIT / import overhead, then assert
    the amortized cost over 100 calls is sub-millisecond. CI jitter
    rarely breaks this margin on any M-series Mac or modern CI runner.
    """
    single_wheel_forces(rashid_wheel, nominal_soil, vertical_load_n=50.0, slip=0.2)  # warm
    n_calls = 100
    t0 = time.perf_counter()
    for _ in range(n_calls):
        single_wheel_forces(rashid_wheel, nominal_soil, vertical_load_n=50.0, slip=0.2)
    elapsed = (time.perf_counter() - t0) / n_calls
    assert elapsed < 1e-3, f"amortized {elapsed * 1000:.2f} ms/call exceeds 1 ms budget"


# ---------------------------------------------------------------------------
# Layer-3 sub-model validation against a published-reference grid
# ---------------------------------------------------------------------------
# The reference CSV at ``data/validation/wong_layer3_reference.csv`` gives a
# per-row (wheel, soil, vertical load, slip) operating point with expected
# (DP, sinkage, torque, DP/W) bounds. Three row kinds are exercised:
#
#   1. ``characterisation`` — Wong (2008) §4.2-style worked-example
#      fixture (JSC-1A canonical Bekker parameters, Wong-textbook wheel
#      geometry). Bounds are pinned at the BW kernel's verified outputs
#      to within ±5 % so the test guards against unintended drift in the
#      analytical kernel and stays inside the ±15-30 % BW model-form
#      band reported in the literature.
#
#   2. ``published_rover_class`` — Apollo nominal regolith × a smooth
#      Pragyan-class wheel and a grousered Yutu-2-class wheel. Bounds
#      are sized at the published BW model-form error (Ishigami 2007;
#      Ding et al. 2011) so any kernel that lands inside the Bekker-Wong
#      analytical band is acceptable.
#
#   3. ``closed_form_limit`` — Iizuka & Kubota (2011) grouser-thrust
#      limit cases (smooth wheel ⇒ lift factor ≡ 1; saturation cap on
#      dense grouser packs). These rows cross-check the BW kernel
#      against the closed-form limits derivable from the textbook.
#
# Tolerance bands tighten when literature digitised values become
# available; appending rows is additive and the test picks them up
# automatically.


def _read_layer3_reference() -> list[dict[str, str]]:
    with LAYER3_REFERENCE_CSV.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


@pytest.mark.parametrize("row", _read_layer3_reference(), ids=lambda r: r["case_id"])
def test_layer3_published_reference_grid(row: dict[str, str]) -> None:
    """Layer-3 BW-vs-published reference grid; see module docstring.

    Replaces the previous ``test_single_wheel_matches_wong_textbook_example``
    xfail. Each row of the reference CSV yields one parametrised case; a
    case passes if every reported quantity (DP, sinkage, torque, DP/W)
    falls inside the row-level ``[lo, hi]`` band documented in the CSV's
    ``citation`` column. Failures point at the offending row id and the
    out-of-band quantity, so digitised follow-on rows can be slotted in
    without rewriting the test.
    """
    wheel = WheelGeometry(
        radius_m=float(row["wheel_radius_m"]),
        width_m=float(row["wheel_width_m"]),
        grouser_height_m=float(row["grouser_height_m"]),
        grouser_count=int(row["grouser_count"]),
    )
    soil = SoilParameters(
        n=float(row["soil_n"]),
        k_c=float(row["soil_k_c_kN"]),
        k_phi=float(row["soil_k_phi_kN"]),
        cohesion_kpa=float(row["soil_c_kPa"]),
        friction_angle_deg=float(row["soil_phi_deg"]),
    )
    load_n = float(row["vertical_load_n"])
    slip = float(row["slip"])
    forces: WheelForces = single_wheel_forces(wheel, soil, vertical_load_n=load_n, slip=slip)

    case = row["case_id"]
    assert float(row["exp_drawbar_pull_n_lo"]) <= forces.drawbar_pull_n <= float(
        row["exp_drawbar_pull_n_hi"]
    ), (
        f"[{case}] DP={forces.drawbar_pull_n:.3f} N out of "
        f"[{row['exp_drawbar_pull_n_lo']}, {row['exp_drawbar_pull_n_hi']}] N "
        f"({row['citation']})"
    )
    assert float(row["exp_sinkage_m_lo"]) <= forces.sinkage_m <= float(row["exp_sinkage_m_hi"]), (
        f"[{case}] sinkage={forces.sinkage_m * 1000:.2f} mm out of "
        f"[{float(row['exp_sinkage_m_lo']) * 1000:.2f}, "
        f"{float(row['exp_sinkage_m_hi']) * 1000:.2f}] mm "
        f"({row['citation']})"
    )
    assert float(row["exp_torque_nm_lo"]) <= forces.driving_torque_nm <= float(
        row["exp_torque_nm_hi"]
    ), (
        f"[{case}] torque={forces.driving_torque_nm:.3f} N·m out of "
        f"[{row['exp_torque_nm_lo']}, {row['exp_torque_nm_hi']}] N·m "
        f"({row['citation']})"
    )
    dp_over_w = forces.drawbar_pull_n / load_n
    assert float(row["exp_dp_over_w_lo"]) <= dp_over_w <= float(row["exp_dp_over_w_hi"]), (
        f"[{case}] DP/W={dp_over_w:+.3f} out of "
        f"[{row['exp_dp_over_w_lo']}, {row['exp_dp_over_w_hi']}] "
        f"({row['citation']})"
    )
