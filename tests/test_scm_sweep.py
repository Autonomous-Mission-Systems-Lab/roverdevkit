"""Unit tests for the SCM-correction SCM sweep design generator and worker.

The design generator is pure NumPy / scipy / pandas and runs in the fast
pytest loop. The worker (:func:`run_one`) is exercised under a chrono+slow
gate so CI without PyChrono still passes.
"""

from __future__ import annotations

import pandas as pd
import pytest

from roverdevkit.terramechanics.pychrono_scm import is_available as _scm_available
from roverdevkit.terramechanics.scm_sweep import (
    CONTINUOUS_BOUNDS,
    DESIGN_COLUMNS,
    GROUSER_COUNTS,
    SOIL_CLASSES,
    build_design,
    run_one,
)

# ---------------------------------------------------------------------------
# Design generator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_runs", [12, 60, 500])
def test_build_design_balances_categorical_buckets(n_runs: int) -> None:
    """Every (soil × grouser) bucket gets the same count ± 1 row."""
    df = build_design(n_runs)
    counts = df.groupby(["soil_class", "grouser_count"]).size()
    assert len(counts) == len(SOIL_CLASSES) * len(GROUSER_COUNTS)
    assert counts.max() - counts.min() <= 1, f"bucket sizes too uneven: {counts.to_dict()}"
    assert counts.sum() == n_runs


def test_build_design_continuous_columns_within_bounds() -> None:
    df = build_design(200)
    for col, (lo, hi) in CONTINUOUS_BOUNDS.items():
        assert df[col].min() >= lo, f"{col} below lower bound"
        assert df[col].max() <= hi, f"{col} above upper bound"


def test_build_design_columns_match_schema() -> None:
    df = build_design(20)
    assert tuple(df.columns) == DESIGN_COLUMNS
    assert df["row_id"].is_unique
    assert df["row_id"].min() == 0
    assert df["row_id"].max() == len(df) - 1


def test_build_design_reproducible() -> None:
    """Same (n_runs, seed) → identical design."""
    a = build_design(50, seed=7)
    b = build_design(50, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_build_design_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        build_design(0)
    with pytest.raises(ValueError):
        build_design(10, soil_classes=())
    with pytest.raises(ValueError):
        build_design(10, grouser_counts=())


# ---------------------------------------------------------------------------
# Worker (chrono+slow)
# ---------------------------------------------------------------------------


@pytest.mark.chrono
@pytest.mark.slow
@pytest.mark.skipif(not _scm_available(), reason="PyChrono not installed")
def test_run_one_returns_expected_keys_and_status() -> None:
    """Smoke: run a single row at the cheapest config; check schema + ok status."""
    from roverdevkit.terramechanics.pychrono_scm import ScmConfig

    fast = ScmConfig(
        time_step_s=1e-3,
        settle_time_s=0.2,
        drive_time_s=0.8,
        terrain_mesh_res_m=0.02,
        average_window_skip=0.5,
    )
    row = build_design(1, seed=42).iloc[0].to_dict()
    out = run_one(row, scm_config=fast)

    expected_design = set(DESIGN_COLUMNS)
    expected_soil = {
        "soil_n",
        "soil_k_c",
        "soil_k_phi",
        "soil_cohesion_kpa",
        "soil_friction_angle_deg",
        "soil_shear_modulus_k_m",
    }
    expected_bw = {"bw_status", "bw_drawbar_pull_n", "bw_torque_nm", "bw_sinkage_m"}
    expected_scm = {
        "scm_status",
        "scm_drawbar_pull_n",
        "scm_torque_nm",
        "scm_sinkage_m",
        "scm_wall_clock_s",
        "scm_fz_residual_n",
        "scm_n_avg_samples",
    }
    missing = (expected_design | expected_soil | expected_bw | expected_scm) - set(out)
    assert not missing, f"missing keys in run_one output: {sorted(missing)}"

    assert out["bw_status"] == "ok"
    assert out["scm_status"] == "ok"
    assert out["scm_wall_clock_s"] > 0


def test_run_one_rejects_incomplete_row() -> None:
    """Missing required keys produce a clear KeyError, not a worker-side trace."""
    with pytest.raises(KeyError):
        run_one({"slip": 0.1})
