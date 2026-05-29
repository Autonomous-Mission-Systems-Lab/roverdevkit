"""Unit tests for the parallel dataset builder (initial baseline-surrogate schema).

These tests use :func:`build_dataset` with ``n_workers=1`` for
reproducibility and to avoid multiprocessing fork/spawn overhead in CI.
A small dedicated parallel-smoke test exercises the spawn path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from roverdevkit.surrogate.dataset import (
    SCHEMA_VERSION,
    DatasetMetadata,
    build_dataset,
    read_parquet,
    read_parquet_metadata,
    write_parquet,
)
from roverdevkit.surrogate.sampling import generate_samples

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_df() -> pd.DataFrame:
    """Shared tiny dataset; one evaluator run per scenario family."""
    samples = generate_samples(n_per_scenario=2, seed=13)
    return build_dataset(samples, n_workers=1, progress=False)


# ---------------------------------------------------------------------------
# Schema: columns and dtypes
# ---------------------------------------------------------------------------


_EXPECTED_META_COLS = {
    "sample_index",
    "split",
    "stratum_id",
    "fidelity",
    "status",
}

_EXPECTED_DESIGN_COLS = {
    "design_wheel_radius_m",
    "design_wheel_width_m",
    "design_grouser_height_m",
    "design_grouser_count",
    "design_n_wheels",
    "design_chassis_mass_kg",
    "design_wheelbase_m",
    "design_solar_area_m2",
    "design_battery_capacity_wh",
    "design_avionics_power_w",
    "design_peak_wheel_torque_nm",
}

_EXPECTED_SCENARIO_COLS = {
    "scenario_family",
    "scenario_name",
    "scenario_latitude_deg",
    "scenario_traverse_distance_m",
    "scenario_terrain_class",
    "scenario_soil_simulant",
    "scenario_mission_duration_earth_days",
    "scenario_max_slope_deg",
    "scenario_operational_duty_cycle",
    "scenario_sun_geometry",
    "scenario_soil_n",
    "scenario_soil_k_c",
    "scenario_soil_k_phi",
    "scenario_soil_cohesion_kpa",
    "scenario_soil_friction_angle_deg",
    "scenario_soil_shear_modulus_k_m",
    "scenario_payload_mass_kg",
    "scenario_payload_power_w",
}

_EXPECTED_METRIC_COLS = {
    "range_km",
    "energy_margin_pct",
    "energy_margin_raw_pct",
    "slope_capability_deg",
    "total_mass_kg",
    "peak_motor_torque_nm",
    "sinkage_max_m",
    "stalled",
}
# thermal_survival was removed at SCHEMA_VERSION = v2; the surrogate no
# longer consumes it (the system-level evaluator still computes it as a
# diagnostic). See data/analytical/SCHEMA.md for the rationale.
_THERMAL_REMOVED_COLS = {"thermal_survival"}


def test_row_count_matches_sample_count(small_df: pd.DataFrame) -> None:
    assert len(small_df) == 8  # 2 per scenario × 4 scenarios


def test_expected_columns_present(small_df: pd.DataFrame) -> None:
    cols = set(small_df.columns)
    for expected in (
        _EXPECTED_META_COLS,
        _EXPECTED_DESIGN_COLS,
        _EXPECTED_SCENARIO_COLS,
        _EXPECTED_METRIC_COLS,
    ):
        missing = expected - cols
        assert not missing, f"missing columns: {missing}"


def test_stat_columns_present(small_df: pd.DataFrame) -> None:
    stat_cols = {c for c in small_df.columns if c.startswith("stat_")}
    # At least 24 stat columns (20 numeric + 3 bool + 1 categorical reason).
    assert len(stat_cols) >= 24, stat_cols


def test_categorical_columns_are_categorical(small_df: pd.DataFrame) -> None:
    for col in [
        "split",
        "scenario_family",
        "scenario_name",
        "scenario_terrain_class",
        "scenario_soil_simulant",
        "scenario_sun_geometry",
        "fidelity",
        "status",
        "stat_terminated_reason",
    ]:
        assert isinstance(small_df[col].dtype, pd.CategoricalDtype), col


def test_design_n_wheels_is_4_or_6(small_df: pd.DataFrame) -> None:
    assert set(small_df["design_n_wheels"].unique()) <= {4, 6}


def test_rows_ordered_by_sample_index(small_df: pd.DataFrame) -> None:
    idx = small_df["sample_index"].to_numpy()
    assert np.all(idx == np.sort(idx))


def test_all_rows_succeeded_on_happy_path(small_df: pd.DataFrame) -> None:
    assert (small_df["status"] == "ok").all()


# ---------------------------------------------------------------------------
# Metric sanity
# ---------------------------------------------------------------------------


def test_metric_ranges_are_physically_plausible(small_df: pd.DataFrame) -> None:
    ok = small_df[small_df["status"] == "ok"]
    assert (ok["range_km"] >= 0).all()
    assert (ok["range_km"] < 1000).all()  # sanity ceiling
    assert (ok["energy_margin_pct"] >= 0).all()
    assert (ok["energy_margin_pct"] <= 100).all()
    assert (ok["slope_capability_deg"] >= 0).all()
    assert (ok["slope_capability_deg"] <= 90).all()
    assert (ok["total_mass_kg"] > 0).all()


def test_stalled_is_boolean(small_df: pd.DataFrame) -> None:
    assert small_df["stalled"].dtype == bool


def test_thermal_survival_not_in_schema(small_df: pd.DataFrame) -> None:
    """v2 schema removes thermal_survival from the dataset (see SCHEMA.md)."""
    for col in _THERMAL_REMOVED_COLS:
        assert col not in small_df.columns, (
            f"{col} should not be in the v2 schema; the surrogate does not "
            "predict thermal until the mass model charges RHU/MLI mass."
        )


def test_stat_columns_are_not_all_nan_on_ok_rows(small_df: pd.DataFrame) -> None:
    ok = small_df[small_df["status"] == "ok"]
    for col in [
        "stat_power_out_mean_w",
        "stat_mobility_power_max_w",
        "stat_soc_final",
    ]:
        assert ok[col].notna().all(), col


# ---------------------------------------------------------------------------
# Parquet round-trip
# ---------------------------------------------------------------------------


def test_parquet_roundtrip_preserves_schema(tmp_path: Path, small_df: pd.DataFrame) -> None:
    meta = DatasetMetadata(
        sampler_seed=13,
        n_per_scenario=2,
        scenario_families=("equatorial_mare_traverse",),
        notes="roundtrip test",
    )
    out_path = tmp_path / "tiny.parquet"
    write_parquet(small_df, out_path, metadata=meta)
    assert out_path.exists()

    loaded = read_parquet(out_path)
    assert len(loaded) == len(small_df)
    assert set(loaded.columns) == set(small_df.columns)

    # Numeric columns equal within tolerance
    for col in ["range_km", "energy_margin_pct", "total_mass_kg"]:
        np.testing.assert_allclose(
            loaded[col].to_numpy(), small_df[col].to_numpy(), rtol=1e-9, atol=0.0
        )


def test_parquet_metadata_written_and_read_back(tmp_path: Path, small_df: pd.DataFrame) -> None:
    meta = DatasetMetadata(
        sampler_seed=13,
        n_per_scenario=2,
        scenario_families=("equatorial_mare_traverse",),
        notes="metadata test",
    )
    out_path = tmp_path / "tiny.parquet"
    write_parquet(small_df, out_path, metadata=meta)
    md = read_parquet_metadata(out_path)
    assert md["schema_version"] == SCHEMA_VERSION
    assert md["sampler_seed"] == "13"
    assert md["notes"] == "metadata test"


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------


def test_build_dataset_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="No samples"):
        build_dataset([], n_workers=1, progress=False)


def test_evaluator_failure_is_recorded_not_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject a failure into evaluate_verbose and check the row is
    kept with status = exception class name and NaN numeric outputs."""
    import roverdevkit.surrogate.dataset as ds_mod

    def boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("injected failure")

    monkeypatch.setattr(ds_mod, "evaluate_verbose", boom)
    samples = generate_samples(n_per_scenario=2, seed=0)
    df = build_dataset(samples[:2], n_workers=1, progress=False)
    assert len(df) == 2
    assert (df["status"] == "RuntimeError").all()
    assert df["range_km"].isna().all()
    assert df["energy_margin_pct"].isna().all()
    # Schema v6: failed rows default to ``stalled=True`` (the
    # safe-conservative side; status != 'ok' filters them out anyway).
    assert (df["stalled"] == True).all()  # noqa: E712


# ---------------------------------------------------------------------------
# Parallel smoke test (spawn context)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_build_dataset_parallel_matches_serial() -> None:
    """Run 4 samples both serially and with 2 workers; expect identical outputs."""
    samples = generate_samples(n_per_scenario=2, seed=0, scenario_names=["crater_rim_survey"])
    serial = build_dataset(samples, n_workers=1, progress=False)
    parallel = build_dataset(samples, n_workers=2, chunksize=2, progress=False)
    assert list(serial["sample_index"]) == list(parallel["sample_index"])
    np.testing.assert_allclose(
        serial["range_km"].to_numpy(), parallel["range_km"].to_numpy(), rtol=1e-9, atol=0.0
    )
    np.testing.assert_allclose(
        serial["total_mass_kg"].to_numpy(),
        parallel["total_mass_kg"].to_numpy(),
        rtol=1e-9,
        atol=0.0,
    )
