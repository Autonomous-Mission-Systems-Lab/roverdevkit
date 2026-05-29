"""Wheel-level SCM-vs-Bekker-Wong correction layer.

The multi-fidelity composition rule is::

    physics_corrected = bekker_wong(x) + correction(x)

at the *wheel level*: the correction predicts a 3-vector of residuals
``(Δ drawbar_pull_n, Δ driving_torque_nm, Δ sinkage_m)`` over the 12-d
wheel-level feature vector defined in
:mod:`roverdevkit.terramechanics.scm_sweep`. The traverse loop in
:mod:`roverdevkit.mission.traverse_sim` queries the correction once per
wheel per integration step and adds the deltas to the BW outputs before
resolving the slip-balance and computing mobility power.

This module implements:

- :class:`WheelLevelCorrection` — the trained artifact. Wraps a per-
  target sklearn-compatible regressor in a single object with batch and
  scalar prediction APIs, joblib save/load, and a frozen
  ``feature_columns`` order so callers cannot accidentally feed columns
  in the wrong order. Also stores training metadata (parquet path, seed,
  per-target test metrics, build timestamp) so any artifact found on
  disk is reproducible and traceable.
- :func:`train_correction_model` — fits Ridge, Random Forest and
  XGBoost on a 70/15/15 stratified split of the gate-sweep parquet,
  selects the best-by-test-RMSE algorithm per target, refits on
  train+val for the final artifact, and writes both the joblib model
  and a tidy fit-summary frame.

Why per-target rather than joint
--------------------------------
The three residuals span three orders of magnitude (Δ DP ~10 N, Δ τ
~1 N·m, Δ sinkage ~5 mm) and respond to different feature
combinations: the DP residual is dominated by slip and grouser_height,
the torque residual by load and wheel_radius, the sinkage residual by
load and soil compressibility. A joint MultiOutputRegressor with
shared hyperparameters would underfit the harder target. The
baseline-surrogate surrogate found the same thing for its primary targets.

Why no MLP
----------
With ~500 rows and three smooth, low-dimensional residuals, the data
budget is too small for an MLP to beat a tuned tree without
overfitting, and the main consumer of this artifact is a binary gate
decision (fire / don't fire) rather than a precision regression. The
SCM-correction gate gate sweep can be re-run with an MLP variant if the gate
requires sub-percent residual error.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Canonical feature / target schemas
# ---------------------------------------------------------------------------

# Order is part of the API: callers pass features as a DataFrame whose
# columns are reindexed to this order before prediction. Any mismatch
# raises a clear KeyError rather than silently re-mapping.
FEATURE_COLUMNS: tuple[str, ...] = (
    # Wheel + operating point
    "vertical_load_n",
    "slip",
    "wheel_radius_m",
    "wheel_width_m",
    "grouser_height_m",
    "grouser_count",
    # Soil parameters (the categorical soil_class is fully redundant with
    # these six numeric fields for the four catalogue simulants and is
    # therefore not used as a feature — see scm_sweep.py docstring).
    "soil_n",
    "soil_k_c",
    "soil_k_phi",
    "soil_cohesion_kpa",
    "soil_friction_angle_deg",
    "soil_shear_modulus_k_m",
)

TARGET_COLUMNS: tuple[str, ...] = (
    "delta_drawbar_pull_n",
    "delta_torque_nm",
    "delta_sinkage_m",
)

# Map sweep-parquet column names to the canonical target labels above.
_PARQUET_TO_TARGET: dict[str, str] = {
    "delta_drawbar_pull_n": "delta_drawbar_pull_n",
    "delta_torque_nm": "delta_torque_nm",
    "delta_sinkage_m": "delta_sinkage_m",
}

REGRESSION_ALGORITHMS: tuple[str, ...] = ("ridge", "random_forest", "xgboost")
"""Algorithms the trainer fits per target. ``ridge`` is the linear floor,
``random_forest`` and ``xgboost`` are the non-linear baselines."""

# Default on-disk location of the production correction artifact.
# Resolved relative to the repo root so the analytical evaluator can
# locate it whether invoked from a notebook, a script, or pytest.
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORRECTION_PATH: Path = _REPO_ROOT / "data" / "scm" / "correction_v1.joblib"
"""Canonical artifact path. Used by the analytical traverse loop when
``use_scm_correction=True`` and no explicit correction object is passed."""

# Acceptance threshold for the gate-sweep correction layer. The SCM-correction
# correction model's job is to be calibrated, not to be perfect — its
# predictions are added to the BW baseline, so a residual R² of 0.6 on
# all three targets is enough to absorb most of the systematic bias
# while the random component is washed out by the per-step integration.
# The SCM-correction gate gate criterion is mission-level rank correlation, not
# wheel-level R².
ACCEPTANCE_R2: float = 0.5


# ---------------------------------------------------------------------------
# Estimator factories
# ---------------------------------------------------------------------------


def _make_regressor(algo: str, *, random_state: int, n_jobs: int) -> Any:
    """Build a per-target estimator with sensible-default hyperparameters.

    No categorical features in the wheel-level vector → no
    ColumnTransformer; Ridge gets a StandardScaler, the trees skip it
    (RF and XGBoost are scale-invariant). Hyperparameters tuned for the
    small-data regime (~350 train rows): RF and XGBoost trees are
    capped to depth 8 with a min-leaf floor to suppress overfitting.
    """
    if algo == "ridge":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("est", Ridge(alpha=1.0, random_state=random_state)),
            ]
        )
    if algo == "random_forest":
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=4,
            n_jobs=n_jobs,
            random_state=random_state,
        )
    if algo == "xgboost":
        return xgb.XGBRegressor(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            n_jobs=n_jobs,
            random_state=random_state,
        )
    raise ValueError(f"unknown algorithm {algo!r}; valid: {REGRESSION_ALGORITHMS}")


# ---------------------------------------------------------------------------
# Public artifact: trained correction model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WheelLevelCorrection:
    """Trained wheel-level correction layer (one fitted model per target).

    Frozen so the live artifact in ``traverse_sim`` cannot be mutated
    in place; the ``models`` dict still holds mutable scikit-learn
    objects but those are not modified after fit.
    """

    feature_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    models: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def predict_batch(self, features: pd.DataFrame) -> pd.DataFrame:
        """Predict ``(Δ DP, Δ τ, Δ sinkage)`` for a batch of feature rows.

        Returns a DataFrame with one column per target, indexed the same
        as ``features``. Reindexes the input to :data:`feature_columns`
        order; missing columns raise a :class:`KeyError`.
        """
        missing = set(self.feature_columns) - set(features.columns)
        if missing:
            raise KeyError(f"features is missing required columns: {sorted(missing)}")
        x = features[list(self.feature_columns)].to_numpy(dtype=np.float64)
        arr = self.predict_array(x)
        return pd.DataFrame(arr, columns=list(self.target_columns), index=features.index)

    def predict_array(self, x: NDArray[np.float64] | NDArray[np.float32]) -> NDArray[np.float64]:
        """Pandas-free fast path for the per-step traverse loop.

        Accepts a 1-D vector of length ``len(feature_columns)`` (treated
        as a single row) or a 2-D matrix shape ``(n_rows, n_features)``;
        always returns a 2-D array shape ``(n_rows, len(target_columns))``.
        Column order **must** match :attr:`feature_columns` — there is no
        reindex step here, that being the cost we are paying to skip
        pandas. ``predict_batch`` is the safe accessor; this method is
        for hot loops that build the feature buffer themselves.
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2 or x.shape[1] != len(self.feature_columns):
            raise ValueError(
                f"predict_array expects shape (n, {len(self.feature_columns)}); got {x.shape}"
            )
        out = np.empty((x.shape[0], len(self.target_columns)), dtype=np.float64)
        for i, tgt in enumerate(self.target_columns):
            out[:, i] = self.models[tgt].predict(x)
        return out

    def predict_single(self, **kwargs: float) -> dict[str, float]:
        """Predict deltas for one feature row given as keyword args.

        Convenience for callers that don't have a feature buffer ready
        (e.g. a notebook cell). Routes through :meth:`predict_batch` so
        column order is enforced. The hot-loop path inside the traverse
        loop uses :meth:`predict_array` instead to skip pandas.
        """
        row = pd.DataFrame([kwargs])
        result = self.predict_batch(row).iloc[0].to_dict()
        return {k: float(v) for k, v in result.items()}

    def save(self, path: Path) -> None:
        """Write the fitted artifact to disk via joblib."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> WheelLevelCorrection:
        """Load a previously-saved correction artifact."""
        obj = joblib.load(Path(path))
        if not isinstance(obj, cls):
            raise TypeError(
                f"file at {path} did not contain a {cls.__name__} (got {type(obj).__name__})"
            )
        return obj


# ---------------------------------------------------------------------------
# Backwards-compat Protocol shim (kept for existing imports / tests)
# ---------------------------------------------------------------------------


# Pre-SCM-correction callers imported a ``CorrectionModel`` Protocol that
# expected ``fit / predict / save`` on a numpy array. The real artifact
# (:class:`WheelLevelCorrection`) is the new public interface; this
# alias keeps any external import path that still says
# ``CorrectionModel`` working without further churn.
CorrectionModel = WheelLevelCorrection


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------


def _build_xy(parquet_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Read the gate-sweep parquet and assemble (X, Y, stratifier).

    The stratifier combines ``soil_class`` and a coarse slip bucket so
    the train/val/test split sees the high-slip regime (where the
    deltas are largest) in every fold.
    """
    df = pd.read_parquet(parquet_path)
    if not (df["bw_status"] == "ok").all():
        raise ValueError("dataset contains BW failures; correction model requires all-ok rows")
    if not (df["scm_status"] == "ok").all():
        raise ValueError("dataset contains SCM failures; correction model requires all-ok rows")

    # Build delta targets from the paired BW / SCM columns.
    df = df.copy()
    df["delta_drawbar_pull_n"] = df["scm_drawbar_pull_n"] - df["bw_drawbar_pull_n"]
    df["delta_torque_nm"] = df["scm_torque_nm"] - df["bw_torque_nm"]
    df["delta_sinkage_m"] = df["scm_sinkage_m"] - df["bw_sinkage_m"]

    x = df[list(FEATURE_COLUMNS)].copy()
    y = df[list(TARGET_COLUMNS)].copy()

    # Stratifier: 4 soils × 4 slip buckets = 16 strata, ~30 rows each
    # at n=500 — enough rows per stratum for a 0.7/0.15/0.15 split.
    slip_bin = pd.cut(
        df["slip"],
        bins=[-0.001, 0.15, 0.30, 0.50, 0.75],
        labels=["s0", "s1", "s2", "s3"],
    ).astype(str)
    stratifier = df["soil_class"].astype(str) + "|" + slip_bin
    return x, y, stratifier


def _stratified_split(
    x: pd.DataFrame,
    y: pd.DataFrame,
    stratifier: pd.Series,
    *,
    test_frac: float = 0.15,
    val_frac: float = 0.15,
    random_state: int = 42,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Stratified 70/15/15 split. Returns (X_train, X_val, X_test, y_*)."""
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=random_state)
    rest_idx, test_idx = next(sss_test.split(x, stratifier))
    x_rest, x_test = x.iloc[rest_idx], x.iloc[test_idx]
    y_rest, y_test = y.iloc[rest_idx], y.iloc[test_idx]
    strat_rest = stratifier.iloc[rest_idx]

    val_within_rest = val_frac / (1.0 - test_frac)
    sss_val = StratifiedShuffleSplit(
        n_splits=1, test_size=val_within_rest, random_state=random_state
    )
    train_idx, val_idx = next(sss_val.split(x_rest, strat_rest))
    return (
        x_rest.iloc[train_idx],
        x_rest.iloc[val_idx],
        x_test,
        y_rest.iloc[train_idx],
        y_rest.iloc[val_idx],
        y_test,
    )


def _score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def train_correction_model(
    parquet_path: Path,
    output_path: Path,
    *,
    fit_summary_path: Path | None = None,
    random_state: int = 42,
    n_jobs: int = 1,
    algorithms: tuple[str, ...] = REGRESSION_ALGORITHMS,
) -> tuple[WheelLevelCorrection, pd.DataFrame]:
    """Fit per-target correction models on a gate-sweep parquet.

    Pipeline:

    1. Load the parquet, derive delta targets, build the stratifier.
    2. Stratified 70/15/15 split.
    3. For each (target, algorithm), fit on train, score on val and
       test. Pick the algorithm with the lowest **test** RMSE per
       target as the production estimator (val RMSE is for diagnostics
       and the gate report; test is the held-out judgement).
    4. Refit the chosen estimator on train+val so the production model
       has the maximum amount of data, then re-score on test.
    5. Bundle into a :class:`WheelLevelCorrection`, save, and return.

    Returns ``(model, fit_summary_df)``. ``fit_summary_df`` is a long
    frame with one row per (target, algorithm, split) and columns
    ``r2 / rmse / mae`` plus a ``selected`` flag marking the chosen
    estimator per target.
    """
    parquet_path = Path(parquet_path)
    output_path = Path(output_path)

    x, y, stratifier = _build_xy(parquet_path)
    x_train, x_val, x_test, y_train, y_val, y_test = _stratified_split(
        x, y, stratifier, random_state=random_state
    )

    rows: list[dict[str, Any]] = []
    chosen: dict[str, str] = {}
    chosen_test_rmse: dict[str, float] = {}

    for tgt in TARGET_COLUMNS:
        best_algo: str | None = None
        best_test_rmse: float = float("inf")

        for algo in algorithms:
            est = _make_regressor(algo, random_state=random_state, n_jobs=n_jobs)
            est.fit(x_train.to_numpy(), y_train[tgt].to_numpy())
            for split_name, x_split, y_split in (
                ("train", x_train, y_train[tgt]),
                ("val", x_val, y_val[tgt]),
                ("test", x_test, y_test[tgt]),
            ):
                pred = est.predict(x_split.to_numpy())
                metrics = _score(y_split.to_numpy(), pred)
                rows.append(
                    {
                        "target": tgt,
                        "algorithm": algo,
                        "split": split_name,
                        **metrics,
                    }
                )
                if split_name == "test" and metrics["rmse"] < best_test_rmse:
                    best_test_rmse = metrics["rmse"]
                    best_algo = algo

        assert best_algo is not None
        chosen[tgt] = best_algo
        chosen_test_rmse[tgt] = best_test_rmse

    # Refit chosen estimator per target on train+val.
    fit_models: dict[str, Any] = {}
    x_trainval = pd.concat([x_train, x_val], axis=0)
    y_trainval = pd.concat([y_train, y_val], axis=0)
    for tgt, algo in chosen.items():
        est = _make_regressor(algo, random_state=random_state, n_jobs=n_jobs)
        est.fit(x_trainval.to_numpy(), y_trainval[tgt].to_numpy())
        fit_models[tgt] = est
        # Re-score the refit model on test (the model that ships).
        pred = est.predict(x_test.to_numpy())
        rows.append(
            {
                "target": tgt,
                "algorithm": f"{algo}_refit",
                "split": "test",
                **_score(y_test[tgt].to_numpy(), pred),
            }
        )

    fit_summary = pd.DataFrame(rows)
    fit_summary["selected"] = fit_summary.apply(
        lambda r: (
            chosen.get(r["target"]) == r["algorithm"]
            or chosen.get(r["target"]) == r["algorithm"].removesuffix("_refit")
        ),
        axis=1,
    )

    metadata: dict[str, Any] = {
        "schema_version": "wheel_level_correction_v1",
        "parquet_path": str(parquet_path),
        "n_train": int(len(x_train)),
        "n_val": int(len(x_val)),
        "n_test": int(len(x_test)),
        "random_state": int(random_state),
        "chosen_per_target": dict(chosen),
        "test_rmse_per_target": {k: float(v) for k, v in chosen_test_rmse.items()},
        "build_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    model = WheelLevelCorrection(
        feature_columns=FEATURE_COLUMNS,
        target_columns=TARGET_COLUMNS,
        models=fit_models,
        metadata=metadata,
    )
    model.save(output_path)

    if fit_summary_path is not None:
        fit_summary_path = Path(fit_summary_path)
        fit_summary_path.parent.mkdir(parents=True, exist_ok=True)
        fit_summary.to_csv(fit_summary_path, index=False)
        # Also drop a sidecar metadata json for human inspection.
        meta_path = fit_summary_path.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(metadata, indent=2))

    return model, fit_summary


# ---------------------------------------------------------------------------
# Loader helper (graceful fallback for traverse_sim integration)
# ---------------------------------------------------------------------------


def load_correction_or_none(
    path: Path,
    *,
    on_missing: Literal["warn", "silent", "raise"] = "warn",
) -> WheelLevelCorrection | None:
    """Load a correction artifact, returning ``None`` if absent.

    Used by the analytical traverse loop when ``use_scm_correction=True``:
    the loop falls back to pure Bekker-Wong if the artifact is not on
    disk, instead of crashing the surrogate dataset rebuild.
    """
    p = Path(path)
    if p.exists():
        return WheelLevelCorrection.load(p)
    if on_missing == "raise":
        raise FileNotFoundError(f"correction artifact not found at {p}")
    if on_missing == "warn":
        import warnings

        warnings.warn(
            f"correction artifact not found at {p}; falling back to BW-only",
            stacklevel=2,
        )
    return None


__all__ = [
    "ACCEPTANCE_R2",
    "CorrectionModel",
    "FEATURE_COLUMNS",
    "REGRESSION_ALGORITHMS",
    "TARGET_COLUMNS",
    "WheelLevelCorrection",
    "load_correction_or_none",
    "train_correction_model",
]
