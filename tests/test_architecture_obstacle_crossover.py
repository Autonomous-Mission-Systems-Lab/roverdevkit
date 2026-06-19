"""Tests for architecture obstacle crossover summary metrics."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "scripts" / "run_architecture_obstacle_crossover.py"


def _load_crossover_module():
    spec = importlib.util.spec_from_file_location("run_architecture_obstacle_crossover", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def crossover():
    return _load_crossover_module()


def test_summary_row_empty_front_uses_nan_not_zero(crossover) -> None:
    row = crossover._summary_row("highland_slope_capability", 0.25, pd.DataFrame())
    assert row["front_empty"] is True
    assert row["n_points"] == 0
    assert pd.isna(row["frac_rocker_bogie"])
    assert pd.isna(row["frac_rigid_4wheel"])


def test_summary_row_all_rocker(crossover) -> None:
    front = pd.DataFrame(
        {
            "mobility_architecture": ["rocker_bogie_6wheel", "rocker_bogie_6wheel"],
            "total_mass_kg": [20.0, 25.0],
            "range_km": [40.0, 50.0],
            "obstacle_capability_m": [0.24, 0.25],
        }
    )
    row = crossover._summary_row("equatorial_mare_traverse", 0.22, front)
    assert row["front_empty"] is False
    assert row["frac_rocker_bogie"] == pytest.approx(1.0)
    assert row["frac_rigid_4wheel"] == pytest.approx(0.0)


def test_default_h_obs_m_capped_at_22cm(crossover) -> None:
    assert crossover.DEFAULT_H_OBS_M[-1] == pytest.approx(0.22)
    assert 0.25 not in crossover.DEFAULT_H_OBS_M
