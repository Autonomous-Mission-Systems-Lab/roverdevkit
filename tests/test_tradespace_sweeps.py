"""Pure-Python unit tests for :mod:`roverdevkit.tradespace.sweeps`.

These tests exercise the grid expansion and backend-pick logic
without touching joblib / xgboost / FastAPI -- they only need
:mod:`roverdevkit.schema` and numpy.
"""

from __future__ import annotations

import numpy as np
import pytest

from roverdevkit.schema import DesignVector
from roverdevkit.tradespace.sweeps import (
    EVALUATOR_AUTO_THRESHOLD,
    EVALUATOR_HARD_LIMIT,
    SURROGATE_HARD_LIMIT,
    SweepAxis,
    SweepResult,
    SweepSpec,
    compute_sensitivity,
    expand_grid,
    pick_backend,
)


def _base_design() -> DesignVector:
    return DesignVector(
        wheel_radius_m=0.10,
        wheel_width_m=0.10,
        grouser_height_m=0.012,
        grouser_count=14,
        n_wheels=6,
        chassis_mass_kg=20.0,
        wheelbase_m=0.6,
        solar_area_m2=0.5,
        battery_capacity_wh=100.0,
        avionics_power_w=15.0,
        peak_wheel_torque_nm=1.5,
    )


def test_sweep_axis_values_endpoints_inclusive() -> None:
    axis = SweepAxis(variable="wheel_radius_m", lo=0.08, hi=0.18, n_points=11)
    vals = axis.values()
    assert vals[0] == pytest.approx(0.08)
    assert vals[-1] == pytest.approx(0.18)
    assert len(vals) == 11


def test_sweep_axis_rejects_n_points_below_two() -> None:
    with pytest.raises(ValueError, match="n_points must be >= 2"):
        SweepAxis("wheel_radius_m", 0.08, 0.18, 1).values()


def test_sweep_axis_rejects_inverted_range() -> None:
    with pytest.raises(ValueError, match="hi must be > lo"):
        SweepAxis("wheel_radius_m", 0.18, 0.08, 5).values()


def test_sweep_spec_rejects_non_primary_target() -> None:
    with pytest.raises(ValueError, match="not a primary regression target"):
        SweepSpec(
            target="not_a_real_metric",
            x_axis=SweepAxis("wheel_radius_m", 0.08, 0.18, 5),
            y_axis=None,
        )


def test_sweep_spec_rejects_duplicate_axis_variables() -> None:
    axis = SweepAxis("wheel_radius_m", 0.08, 0.18, 5)
    with pytest.raises(ValueError, match="must sweep different variables"):
        SweepSpec(target="range_km", x_axis=axis, y_axis=axis)


def test_sweep_spec_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="not in"):
        SweepSpec(
            target="range_km",
            x_axis=SweepAxis("wheel_radius_m", 0.08, 0.18, 5),
            y_axis=None,
            backend="cuda",  # type: ignore[arg-type]
        )


def test_expand_grid_1d_overrides_just_x() -> None:
    spec = SweepSpec(
        target="range_km",
        x_axis=SweepAxis("wheel_radius_m", 0.08, 0.18, 5),
        y_axis=None,
    )
    designs = expand_grid(spec, _base_design())
    assert len(designs) == 5
    radii = [d.wheel_radius_m for d in designs]
    assert radii[0] == pytest.approx(0.08)
    assert radii[-1] == pytest.approx(0.18)
    # All other dims unchanged
    for d in designs:
        assert d.wheel_width_m == pytest.approx(0.10)
        assert d.solar_area_m2 == pytest.approx(0.5)


def test_expand_grid_2d_row_major_y_outer_x_inner() -> None:
    spec = SweepSpec(
        target="range_km",
        x_axis=SweepAxis("wheel_radius_m", 0.08, 0.18, 3),
        y_axis=SweepAxis("solar_area_m2", 0.4, 0.8, 2),
    )
    designs = expand_grid(spec, _base_design())
    assert len(designs) == 6
    # Row-major: first three share y[0], next three share y[1].
    ys = [d.solar_area_m2 for d in designs]
    assert ys[:3] == pytest.approx([0.4, 0.4, 0.4])
    assert ys[3:] == pytest.approx([0.8, 0.8, 0.8])
    xs = [d.wheel_radius_m for d in designs]
    np.testing.assert_allclose(xs[:3], [0.08, 0.13, 0.18])
    np.testing.assert_allclose(xs[3:], [0.08, 0.13, 0.18])


def test_expand_grid_rounds_integer_variable() -> None:
    spec = SweepSpec(
        target="range_km",
        x_axis=SweepAxis("grouser_count", 0.0, 24.0, 5),
        y_axis=None,
    )
    designs = expand_grid(spec, _base_design())
    counts = [d.grouser_count for d in designs]
    # Linear grid is [0, 6, 12, 18, 24]; all integers already.
    assert counts == [0, 6, 12, 18, 24]


def test_pick_backend_auto_uses_evaluator_below_threshold() -> None:
    spec = SweepSpec(
        target="range_km",
        x_axis=SweepAxis("wheel_radius_m", 0.08, 0.18, EVALUATOR_AUTO_THRESHOLD),
        y_axis=None,
    )
    assert pick_backend(spec) == "evaluator"


def test_pick_backend_auto_promotes_to_surrogate_above_threshold() -> None:
    n = EVALUATOR_AUTO_THRESHOLD + 1
    spec = SweepSpec(
        target="range_km",
        x_axis=SweepAxis("wheel_radius_m", 0.08, 0.18, n),
        y_axis=None,
    )
    assert pick_backend(spec) == "surrogate"


def test_pick_backend_explicit_evaluator_hard_limit() -> None:
    n = EVALUATOR_HARD_LIMIT + 1
    spec = SweepSpec(
        target="range_km",
        x_axis=SweepAxis("wheel_radius_m", 0.08, 0.18, n),
        y_axis=None,
        backend="evaluator",
    )
    with pytest.raises(ValueError, match="evaluator hard limit"):
        pick_backend(spec)


# ---------------------------------------------------------------------------
# compute_sensitivity
#
# These tests build SweepResult objects directly with synthetic z grids so
# we can pin down the spread numerics without running the evaluator.
# ---------------------------------------------------------------------------


def _make_result(
    z: np.ndarray,
    *,
    x_n: int,
    y_n: int | None,
) -> SweepResult:
    """Wrap a precomputed ``z`` array in a SweepResult for sensitivity tests."""
    x_axis = SweepAxis("wheel_radius_m", 0.08, 0.18, x_n)
    y_axis = (
        None if y_n is None else SweepAxis("solar_area_m2", 0.4, 0.8, y_n)
    )
    spec = SweepSpec(target="range_km", x_axis=x_axis, y_axis=y_axis)
    return SweepResult(
        spec=spec,
        x_values=x_axis.values(),
        y_values=None if y_axis is None else y_axis.values(),
        z_values=z,
        backend_used="evaluator",
        elapsed_s=0.0,
    )


def test_compute_sensitivity_1d_total_spread_and_relative() -> None:
    z = np.array([10.0, 12.0, 15.0, 18.0, 20.0])
    sens = compute_sensitivity(_make_result(z, x_n=5, y_n=None))
    assert sens.total_spread == pytest.approx(10.0)
    assert sens.relative_spread == pytest.approx(10.0 / 20.0)
    assert sens.axis_spread_x == pytest.approx(10.0)
    assert sens.axis_spread_y is None


def test_compute_sensitivity_constant_grid_returns_zero_relative_spread() -> None:
    # All-NaN guard sits on top, but a flat finite grid is the more
    # interesting "metric saturated" branch that drives the UI hint.
    z = np.full((4, 5), 3.7)
    sens = compute_sensitivity(_make_result(z, x_n=5, y_n=4))
    assert sens.total_spread == pytest.approx(0.0)
    assert sens.relative_spread == pytest.approx(0.0)
    assert sens.axis_spread_x == pytest.approx(0.0)
    assert sens.axis_spread_y == pytest.approx(0.0)


def test_compute_sensitivity_all_nan_grid_zeroed_safely() -> None:
    z = np.full((3, 4), np.nan)
    sens = compute_sensitivity(_make_result(z, x_n=4, y_n=3))
    assert sens.total_spread == 0.0
    assert sens.relative_spread == 0.0
    assert sens.axis_spread_x == 0.0
    assert sens.axis_spread_y == 0.0


def test_compute_sensitivity_2d_x_dominated_grid() -> None:
    # Each row varies strongly with column index (x), but rows differ
    # only by a small additive shift (weak y dependence). Sensitivity
    # along x should be ~10x sensitivity along y.
    base_x = np.array([0.0, 5.0, 10.0])  # spread along x = 10
    rows = np.stack([base_x, base_x + 1.0])  # spread along y at fixed x = 1
    sens = compute_sensitivity(_make_result(rows, x_n=3, y_n=2))
    assert sens.axis_spread_x == pytest.approx(10.0)
    assert sens.axis_spread_y == pytest.approx(1.0)
    # total spread spans both effects: 0 -> 11
    assert sens.total_spread == pytest.approx(11.0)


def test_pick_backend_explicit_surrogate_hard_limit() -> None:
    # 200 × 201 = 40_200 > SURROGATE_HARD_LIMIT (40_000).
    spec = SweepSpec(
        target="range_km",
        x_axis=SweepAxis("wheel_radius_m", 0.08, 0.18, 200),
        y_axis=SweepAxis("solar_area_m2", 0.4, 0.8, 201),
        backend="surrogate",
    )
    assert spec.n_cells() > SURROGATE_HARD_LIMIT
    with pytest.raises(ValueError, match="surrogate hard limit"):
        pick_backend(spec)
