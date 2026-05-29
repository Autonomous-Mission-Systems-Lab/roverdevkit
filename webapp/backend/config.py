"""Backend configuration: artifact paths, CORS origins, dataset version.

All paths default to the in-repo surrogate-training artifacts so the backend works
out of the box from a fresh clone after running the surrogate-training pipeline.
Each value can be overridden via environment variable so the same
container image can be repointed at a remote object store / mounted
volume in deployment without code changes.

Environment variables
---------------------
``ROVERDEVKIT_QUANTILE_BUNDLES``
    Path to ``quantile_bundles.joblib`` (calibrated quantile XGB heads).
    Default: ``reports/surrogate_v9/quantile_bundles.joblib`` — the v9
    recalibration on lhs_v9.parquet after scientific payload was
    promoted from a per-rover ``chassis_mass_kg`` convention to two
    explicit mission-requirement inputs (``payload_mass_kg`` /
    ``payload_power_w``), each an LHS feature uniform on [0, 30]. With
    the v9 calibration, the Mission-Inputs panel's payload sliders stay
    on the surrogate path with calibrated 90 % PIs across the full
    override range.
``ROVERDEVKIT_TUNED_PARAMS``
    Path to ``tuned_best_params.json`` (tuned XGB hyperparameters).
    Currently informational only; reserved for later steps that may
    need to refit. Default:
    ``reports/tuned_v9/tuned_best_params.json`` (50-trial Optuna
    sweep on the v9 dataset for the four primary regressors and the
    ``stalled`` classifier; re-run for v9 because the input
    dimensionality changed from 25 to 27 columns).
``ROVERDEVKIT_DATASET_VERSION``
    Dataset version label echoed in ``/version``. Default ``v9``.
``ROVERDEVKIT_CORS_ORIGINS``
    Comma-separated allow-list. Defaults to the Vite dev server.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    """Resolved backend configuration. Built once via :func:`get_settings`."""

    quantile_bundles_path: Path
    tuned_params_path: Path
    dataset_version: str
    cors_origins: tuple[str, ...]
    repo_root: Path

    @property
    def artifacts_present(self) -> bool:
        """True iff the surrogate artifact exists on disk."""
        return self.quantile_bundles_path.exists()


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser().resolve() if raw else default


def _env_csv(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.environ.get(name)
    if not raw:
        return default
    return tuple(x.strip() for x in raw.split(",") if x.strip())


def get_settings() -> Settings:
    """Build a :class:`Settings` object from process env + repo defaults.

    Called once on app startup and re-resolved on each call so tests
    can monkey-patch via ``os.environ`` between invocations.
    """
    return Settings(
        quantile_bundles_path=_env_path(
            "ROVERDEVKIT_QUANTILE_BUNDLES",
            REPO_ROOT / "reports" / "surrogate_v9" / "quantile_bundles.joblib",
        ),
        tuned_params_path=_env_path(
            "ROVERDEVKIT_TUNED_PARAMS",
            REPO_ROOT / "reports" / "tuned_v9" / "tuned_best_params.json",
        ),
        dataset_version=os.environ.get("ROVERDEVKIT_DATASET_VERSION", "v9"),
        cors_origins=_env_csv(
            "ROVERDEVKIT_CORS_ORIGINS",
            ("http://localhost:5173", "http://127.0.0.1:5173"),
        ),
        repo_root=REPO_ROOT,
    )
