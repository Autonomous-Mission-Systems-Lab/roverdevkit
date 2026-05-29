"""Unit tests for the SCM-correction wheel-level SCM correction layer.

Uses a small synthetic gate-sweep parquet built from the real
:func:`roverdevkit.terramechanics.scm_sweep.build_design` plus mocked
BW / SCM columns. Lives in the fast pytest loop — no PyChrono required.
"""

from __future__ import annotations

import numpy as np
import pytest

from roverdevkit.terramechanics.correction_model import (
    FEATURE_COLUMNS,
    REGRESSION_ALGORITHMS,
    TARGET_COLUMNS,
    WheelLevelCorrection,
    load_correction_or_none,
    train_correction_model,
)
from roverdevkit.terramechanics.scm_sweep import build_design


@pytest.fixture(scope="module")
def synthetic_parquet(tmp_path_factory) -> tuple:
    """Synthetic gate-sweep parquet with a learnable BW→SCM gap.

    Builds a 240-row design (4 soils × 3 grousers × 20 = 240, a
    multiple of the 12-bucket stratifier so the ~70/15/15 splits stay
    balanced) and fabricates BW / SCM outputs whose **delta** is a
    smooth linear-with-noise function of slip and grouser_height. The
    correction model should recover R² > 0.5 on this signal — well
    above what would happen by chance — without requiring the heavy
    PyChrono SCM driver to run.
    """
    rng = np.random.default_rng(0)
    df = build_design(240, seed=0)

    # Mocked soil parameters: distinct constant per soil class so the
    # numeric fields encode the categorical, matching the real catalogue
    # geometry that scm_sweep.run_one would produce.
    soil_table = {
        "Apollo_regolith_nominal": (1.10, 0.14, 820, 0.17, 35.0, 0.018),
        "JSC-1A": (1.00, 0.90, 1500, 1.00, 31.0, 0.020),
        "GRC-1": (0.90, 0.05, 600, 0.20, 30.0, 0.015),
        "FJS-1": (0.80, 0.30, 1100, 0.50, 33.0, 0.018),
    }
    keys = (
        "soil_n",
        "soil_k_c",
        "soil_k_phi",
        "soil_cohesion_kpa",
        "soil_friction_angle_deg",
        "soil_shear_modulus_k_m",
    )
    for idx, k in enumerate(keys):
        df[k] = df["soil_class"].map(lambda s, i=idx: soil_table[s][i])

    # Fabricate BW + SCM columns. BW is a simple proxy of load × slip;
    # SCM = BW + (signal as function of slip, grouser_height) + noise.
    bw_dp = df["vertical_load_n"] * df["slip"] * 0.10
    bw_t = df["vertical_load_n"] * df["wheel_radius_m"] * 0.20
    bw_z = df["vertical_load_n"] * 0.0001

    signal_dp = 50.0 * df["slip"] * df["grouser_height_m"] / 0.020 + 5.0 * df["slip"]
    signal_t = 1.5 * df["slip"] + 2.0 * df["grouser_height_m"] / 0.020
    signal_z = -0.005 * df["slip"]

    noise_dp = rng.normal(0.0, 1.0, len(df))
    noise_t = rng.normal(0.0, 0.05, len(df))
    noise_z = rng.normal(0.0, 0.0005, len(df))

    df["bw_drawbar_pull_n"] = bw_dp
    df["bw_torque_nm"] = bw_t
    df["bw_sinkage_m"] = bw_z
    df["scm_drawbar_pull_n"] = bw_dp + signal_dp + noise_dp
    df["scm_torque_nm"] = bw_t + signal_t + noise_t
    df["scm_sinkage_m"] = bw_z + signal_z + noise_z

    df["bw_status"] = "ok"
    df["scm_status"] = "ok"

    path = tmp_path_factory.mktemp("scm_corr") / "synthetic.parquet"
    df.to_parquet(path, index=False)
    return path, df


def test_train_correction_model_meets_floor_on_synthetic(synthetic_parquet, tmp_path) -> None:
    """End-to-end fit on synthetic data: every target clears R² > 0.5."""
    path, _ = synthetic_parquet
    out_joblib = tmp_path / "correction.joblib"
    out_csv = tmp_path / "fit_summary.csv"
    model, fit_summary = train_correction_model(
        path, out_joblib, fit_summary_path=out_csv, random_state=0
    )

    assert out_joblib.exists()
    assert out_csv.exists()
    assert (out_csv.with_suffix(".meta.json")).exists()

    chosen = model.metadata["chosen_per_target"]
    for tgt in TARGET_COLUMNS:
        assert tgt in chosen
        assert chosen[tgt] in REGRESSION_ALGORITHMS

    refit_rows = fit_summary[
        (fit_summary["split"] == "test") & fit_summary["algorithm"].str.endswith("_refit")
    ]
    assert len(refit_rows) == len(TARGET_COLUMNS)
    assert (refit_rows["r2"] > 0.5).all(), refit_rows


def test_predict_batch_columns_and_shape(synthetic_parquet, tmp_path) -> None:
    path, df = synthetic_parquet
    out_joblib = tmp_path / "correction.joblib"
    model, _ = train_correction_model(path, out_joblib, random_state=0)

    sample = df[list(FEATURE_COLUMNS)].iloc[:8].copy()
    pred = model.predict_batch(sample)
    assert tuple(pred.columns) == TARGET_COLUMNS
    assert pred.shape == (8, len(TARGET_COLUMNS))
    assert pred.notna().all().all()


def test_predict_batch_rejects_missing_features(synthetic_parquet, tmp_path) -> None:
    path, df = synthetic_parquet
    out_joblib = tmp_path / "correction.joblib"
    model, _ = train_correction_model(path, out_joblib, random_state=0)

    sample = df[list(FEATURE_COLUMNS)].iloc[:4].drop(columns=["slip"])
    with pytest.raises(KeyError, match="slip"):
        model.predict_batch(sample)


def test_predict_single_matches_predict_batch(synthetic_parquet, tmp_path) -> None:
    path, df = synthetic_parquet
    out_joblib = tmp_path / "correction.joblib"
    model, _ = train_correction_model(path, out_joblib, random_state=0)

    row = df[list(FEATURE_COLUMNS)].iloc[0]
    batch_pred = model.predict_batch(row.to_frame().T).iloc[0].to_dict()
    single_pred = model.predict_single(**row.to_dict())

    for tgt in TARGET_COLUMNS:
        assert single_pred[tgt] == pytest.approx(batch_pred[tgt], rel=1e-9, abs=1e-12)


def test_save_load_round_trip(synthetic_parquet, tmp_path) -> None:
    path, df = synthetic_parquet
    out_joblib = tmp_path / "correction.joblib"
    model, _ = train_correction_model(path, out_joblib, random_state=0)

    reloaded = WheelLevelCorrection.load(out_joblib)
    sample = df[list(FEATURE_COLUMNS)].iloc[:6].copy()

    p0 = model.predict_batch(sample).to_numpy()
    p1 = reloaded.predict_batch(sample).to_numpy()
    np.testing.assert_array_equal(p0, p1)
    assert reloaded.feature_columns == model.feature_columns
    assert reloaded.target_columns == model.target_columns


def test_load_correction_or_none_missing(tmp_path) -> None:
    p = tmp_path / "does_not_exist.joblib"
    # Default mode is 'warn'; suppressing here since we explicitly assert.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert load_correction_or_none(p) is None
    with pytest.raises(FileNotFoundError):
        load_correction_or_none(p, on_missing="raise")


def test_train_rejects_dataset_with_failed_rows(tmp_path) -> None:
    """A parquet with any BW / SCM failure must refuse to train."""
    df = build_design(48, seed=0)
    for k, v in {
        "soil_n": 1.0,
        "soil_k_c": 0.1,
        "soil_k_phi": 1000.0,
        "soil_cohesion_kpa": 0.5,
        "soil_friction_angle_deg": 32.0,
        "soil_shear_modulus_k_m": 0.018,
        "bw_drawbar_pull_n": 1.0,
        "bw_torque_nm": 0.1,
        "bw_sinkage_m": 0.001,
        "scm_drawbar_pull_n": 2.0,
        "scm_torque_nm": 0.2,
        "scm_sinkage_m": 0.002,
        "bw_status": "ok",
        "scm_status": "ok",
    }.items():
        df[k] = v
    df.loc[0, "scm_status"] = "fail"

    bad = tmp_path / "bad.parquet"
    df.to_parquet(bad, index=False)
    with pytest.raises(ValueError, match="SCM failures"):
        train_correction_model(bad, tmp_path / "out.joblib")
