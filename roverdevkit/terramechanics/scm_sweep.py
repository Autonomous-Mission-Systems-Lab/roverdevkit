"""SCM-correction SCM single-wheel sweep: design generation + per-row worker.

Two thin, testable functions consumed by ``scripts/run_scm_sweep.py``:

- :func:`build_design` — stratified-categorical Latin-hypercube design
  over the 12-d wheel-level feature space defined in the §6 SCM-correction
  composition workflow. Continuous LHS over the six
  numeric wheel/operating parameters; balanced categorical assignment of
  ``soil_class`` and ``grouser_count_class`` so every (soil × grouser)
  bucket gets a roughly equal number of rows. No PyChrono required —
  importable in the fast pytest loop.

- :func:`run_one` — paired Bekker-Wong + PyChrono SCM evaluation of one
  design row. PyChrono is imported **inside** the function so neither
  this module nor any caller pays the ~350 ms OpenMP / Chrono startup
  unless they actually run a row. Returns a flat dict suitable for
  appending to a Parquet store.

Why it lives in the package, not in ``scripts/``
------------------------------------------------
The design generator is unit-testable and may be reused by
notebooks (e.g. for visualising the gate sweep coverage). Putting it
inside ``roverdevkit.terramechanics`` keeps the test surface clean and
lets ``scripts/run_scm_sweep.py`` stay thin (CLI + parallel pool +
Parquet I/O only).

Why ``run_one`` defers the PyChrono import
------------------------------------------
``import pychrono`` triggers an OpenMP preload shim (see
``pychrono_scm.py`` module docstring) that adds ~350 ms to package
init. Analytical-only consumers (NSGA-II, surrogate fits, evaluator
validation) must not pay this cost. The deferral pattern is the same
trick used in ``scripts/run_baselines.py`` for heavy ML imports.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import qmc

# ---------------------------------------------------------------------------
# Constants for the wheel-level SCM correction composition workflow
# ---------------------------------------------------------------------------

# Continuous LHS bounds. Match the v3 design schema where the bound has a
# direct schema analog (wheel_radius_m, wheel_width_m, grouser_height_m);
# vertical_load_n is derived from the chassis-mass × gravity / n_wheels
# envelope with margin for slope projection and dynamic loading.
CONTINUOUS_BOUNDS: dict[str, tuple[float, float]] = {
    "vertical_load_n": (3.0, 80.0),
    "slip": (0.05, 0.70),
    "wheel_radius_m": (0.05, 0.20),
    "wheel_width_m": (0.03, 0.20),
    "grouser_height_m": (0.0, 0.020),
    # grouser_count_continuous is rounded to grouser_count (see below) but
    # carrying it here lets the LHS sampler treat it as a continuous axis
    # rather than as a fully stratified categorical, giving finer coverage
    # of the integer levels at the expense of perfect balance.
}

# Stratified categorical levels. Soil simulants pulled from
# ``data/soil_simulants.csv`` (see ``roverdevkit.terramechanics.soils``).
# The four chosen here cover the catalogue's range of (n, k_c, k_phi,
# cohesion, friction) without redundancy; the three "_dense" / "_loose"
# bound entries are reserved for the error-budget sensitivity sweep.
SOIL_CLASSES: tuple[str, ...] = (
    "Apollo_regolith_nominal",
    "JSC-1A",
    "GRC-1",
    "FJS-1",
)

# Integer grouser counts. 0 = no grousers, 12 = typical lunar micro-rover
# (Yutu-2, Pragyan), 18 = high-traction case (Rashid-1 has 18). Stratified
# rather than continuous so each level is well-represented even at small n.
GROUSER_COUNTS: tuple[int, ...] = (0, 12, 18)

# Output column order: design columns, then BW + SCM result columns. Kept
# as a tuple so the consuming script can validate the parquet schema after
# build.
DESIGN_COLUMNS: tuple[str, ...] = (
    "row_id",
    *CONTINUOUS_BOUNDS.keys(),
    "soil_class",
    "grouser_count",
)


# ---------------------------------------------------------------------------
# Design generator
# ---------------------------------------------------------------------------


def build_design(
    n_runs: int,
    *,
    seed: int = 42,
    bounds: dict[str, tuple[float, float]] | None = None,
    soil_classes: tuple[str, ...] = SOIL_CLASSES,
    grouser_counts: tuple[int, ...] = GROUSER_COUNTS,
) -> pd.DataFrame:
    """Build a stratified-categorical LHS design for the SCM sweep.

    Parameters
    ----------
    n_runs
        Total rows in the design. The (soil × grouser) categorical buckets
        are populated as evenly as possible: each bucket gets either
        ``n_runs // n_buckets`` or one more row.
    seed
        RNG seed for both the LHS sampler and the post-hoc bucket
        permutation. Two builds with the same ``(n_runs, seed)`` produce
        identical designs.
    bounds, soil_classes, grouser_counts
        Override the module-level defaults (mostly useful for tests).

    Returns
    -------
    pandas.DataFrame
        One row per design point. Columns: ``row_id`` plus every key in
        ``bounds``, plus ``soil_class`` and ``grouser_count``. Continuous
        columns are float64; ``soil_class`` is string; ``grouser_count``
        is int.

    Notes
    -----
    The (soil × grouser) labels are assigned independently of the
    continuous LHS columns. This decoupling means the marginal LHS over
    the continuous space is preserved (you still get good 6-d coverage)
    while the categorical buckets are balanced. The row order is then
    randomised so consumers using a chunked write can checkpoint without
    biasing the partial results.
    """
    if n_runs <= 0:
        raise ValueError("n_runs must be positive")
    if not soil_classes or not grouser_counts:
        raise ValueError("soil_classes and grouser_counts must be non-empty")

    bounds = dict(bounds or CONTINUOUS_BOUNDS)
    cols = list(bounds.keys())
    lo = np.array([bounds[c][0] for c in cols], dtype=float)
    hi = np.array([bounds[c][1] for c in cols], dtype=float)

    sampler = qmc.LatinHypercube(d=len(cols), seed=seed)
    unit = sampler.random(n=n_runs)
    scaled = qmc.scale(unit, lo, hi)
    df = pd.DataFrame(scaled, columns=cols)

    # Balanced (soil × grouser) categorical assignment.
    n_soil = len(soil_classes)
    n_grouser = len(grouser_counts)
    n_buckets = n_soil * n_grouser
    base = n_runs // n_buckets
    extra = n_runs % n_buckets

    soil_labels: list[str] = []
    grouser_labels: list[int] = []
    for bucket_idx in range(n_buckets):
        size = base + (1 if bucket_idx < extra else 0)
        s_idx = bucket_idx // n_grouser
        g_idx = bucket_idx % n_grouser
        soil_labels.extend([soil_classes[s_idx]] * size)
        grouser_labels.extend([grouser_counts[g_idx]] * size)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_runs)
    df["soil_class"] = [soil_labels[i] for i in perm]
    df["grouser_count"] = [grouser_labels[i] for i in perm]
    df.insert(0, "row_id", np.arange(n_runs, dtype=np.int64))

    return df[list(DESIGN_COLUMNS)]


# ---------------------------------------------------------------------------
# Per-row worker
# ---------------------------------------------------------------------------


# Required keys in a row dict passed to :func:`run_one`. Validated up-front
# so a typo in a CLI override or notebook usage produces a clear error
# instead of an opaque KeyError inside the worker pool.
_REQUIRED_ROW_KEYS: frozenset[str] = frozenset(DESIGN_COLUMNS)


def run_one(
    row: dict[str, Any],
    *,
    scm_config: Any | None = None,
) -> dict[str, Any]:
    """Run one Bekker-Wong + PyChrono SCM evaluation on a design row.

    Imports PyChrono lazily so the caller (and pytest collection) does
    not pay startup unless the function actually executes.

    Parameters
    ----------
    row
        Mapping with at least the keys in :data:`DESIGN_COLUMNS`.
    scm_config
        Optional :class:`pychrono_scm.ScmConfig`. If ``None``, the
        pychrono_scm default is used (production fidelity, δ=15 mm mesh,
        0.3 s settle + 1.5 s drive).

    Returns
    -------
    dict
        The input ``row`` extended with:

        - ``bw_status`` ∈ {``"ok"``, ``"fail"``} and ``bw_drawbar_pull_n``,
          ``bw_torque_nm``, ``bw_sinkage_m`` (NaN on failure).
        - ``scm_status`` ∈ {``"ok"``, ``"fail"``} and the same three
          quantities under ``scm_*`` plus diagnostic
          ``scm_wall_clock_s``, ``scm_fz_residual_n``,
          ``scm_n_avg_samples``.
        - ``bw_error`` / ``scm_error`` (only present on failure).

    The function never raises; failures are recorded in the row so the
    pool driver can checkpoint partial progress.
    """
    missing = _REQUIRED_ROW_KEYS - set(row)
    if missing:
        raise KeyError(f"row is missing required keys: {sorted(missing)}")

    from roverdevkit.terramechanics.bekker_wong import (
        WheelGeometry,
        single_wheel_forces,
    )
    from roverdevkit.terramechanics.pychrono_scm import (
        ScmConfig,
        single_wheel_forces_scm,
    )
    from roverdevkit.terramechanics.soils import get_soil_parameters

    out = dict(row)
    soil = get_soil_parameters(str(row["soil_class"]))
    wheel = WheelGeometry(
        radius_m=float(row["wheel_radius_m"]),
        width_m=float(row["wheel_width_m"]),
        grouser_height_m=float(row["grouser_height_m"]),
        grouser_count=int(row["grouser_count"]),
    )
    load = float(row["vertical_load_n"])
    slip = float(row["slip"])

    out["soil_n"] = soil.n
    out["soil_k_c"] = soil.k_c
    out["soil_k_phi"] = soil.k_phi
    out["soil_cohesion_kpa"] = soil.cohesion_kpa
    out["soil_friction_angle_deg"] = soil.friction_angle_deg
    out["soil_shear_modulus_k_m"] = soil.shear_modulus_k_m

    try:
        bw = single_wheel_forces(wheel, soil, load, slip)
        out["bw_status"] = "ok"
        out["bw_drawbar_pull_n"] = float(bw.drawbar_pull_n)
        out["bw_torque_nm"] = float(bw.driving_torque_nm)
        out["bw_sinkage_m"] = float(bw.sinkage_m)
    except Exception as exc:
        out["bw_status"] = "fail"
        out["bw_error"] = str(exc)[:240]
        out["bw_drawbar_pull_n"] = float("nan")
        out["bw_torque_nm"] = float("nan")
        out["bw_sinkage_m"] = float("nan")

    cfg = scm_config or ScmConfig()
    telemetry: dict[str, float] = {}
    try:
        scm = single_wheel_forces_scm(wheel, soil, load, slip, config=cfg, telemetry=telemetry)
        out["scm_status"] = "ok"
        out["scm_drawbar_pull_n"] = float(scm.drawbar_pull_n)
        out["scm_torque_nm"] = float(scm.driving_torque_nm)
        out["scm_sinkage_m"] = float(scm.sinkage_m)
        out["scm_wall_clock_s"] = float(telemetry.get("wall_clock_s", float("nan")))
        out["scm_fz_residual_n"] = float(telemetry.get("fz_residual_n", float("nan")))
        out["scm_n_avg_samples"] = float(telemetry.get("n_samples", float("nan")))
    except Exception as exc:
        out["scm_status"] = "fail"
        out["scm_error"] = str(exc)[:240]
        out["scm_drawbar_pull_n"] = float("nan")
        out["scm_torque_nm"] = float("nan")
        out["scm_sinkage_m"] = float("nan")
        out["scm_wall_clock_s"] = float("nan")
        out["scm_fz_residual_n"] = float("nan")
        out["scm_n_avg_samples"] = float("nan")

    return out


__all__ = [
    "CONTINUOUS_BOUNDS",
    "DESIGN_COLUMNS",
    "GROUSER_COUNTS",
    "SOIL_CLASSES",
    "build_design",
    "run_one",
]
