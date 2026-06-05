"""Experiment-vs-model comparison for the Layer-3 terramechanics kernel.

Compares the analytical Bekker-Wong (BW) single-wheel kernel against
*measured* single-wheel drawbar-pull / sinkage / torque from published
planetary-rover terramechanics experiments.

This is the experimental anchor for Layer 3. Unlike
``data/validation/wong_layer3_reference.csv`` (which holds model-form
tolerance *bands*), this module consumes point measurements digitised from
the source figures and reports per-point residuals and percentage errors.

Data source
-----------
``data/validation/single_wheel_experiments.csv`` is a digitisation
worksheet: every row carries the verified operating point (wheel geometry,
vertical load, slip) plus provenance (``source``, ``citation``), and the
``meas_drawbar_pull_n`` / ``meas_sinkage_m`` / ``meas_torque_nm`` columns
are filled in from the published figures. Rows whose measured columns are
blank (``status = pending_digitisation``) are carried through so model
predictions can still be produced, but contribute no error statistics.

The soil Bekker parameters are resolved by simulant name from
``data/soil_simulants.csv``; the Janosi-Hanamoto shear modulus K (absent
from that catalogue) is taken from the worksheet ``soil_shear_modulus_k_m``
column.

"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from roverdevkit.terramechanics.bekker_wong import (
    SoilParameters,
    WheelGeometry,
    single_wheel_forces,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENTS_CSV = (
    _REPO_ROOT / "data" / "validation" / "single_wheel_experiments.csv"
)
DEFAULT_SIMULANTS_CSV = _REPO_ROOT / "data" / "soil_simulants.csv"


@dataclass(frozen=True)
class ExperimentPoint:
    """One measured single-wheel operating point with its model inputs."""

    source: str
    case_id: str
    wheel: WheelGeometry
    soil: SoilParameters
    soil_simulant: str
    vertical_load_n: float
    slip: float
    meas_drawbar_pull_n: float  # NaN until digitised
    meas_sinkage_m: float
    meas_torque_nm: float
    status: str
    citation: str
    notes: str


def _to_float(value: str | None) -> float:
    """Parse a CSV cell to float, mapping blanks to NaN."""
    if value is None:
        return math.nan
    text = value.strip()
    if not text:
        return math.nan
    return float(text)


def load_simulant_bekker_params(
    simulants_csv: Path = DEFAULT_SIMULANTS_CSV,
) -> dict[str, dict[str, float]]:
    """Load Bekker / Mohr-Coulomb parameters keyed by simulant name."""
    params: dict[str, dict[str, float]] = {}
    with Path(simulants_csv).open(newline="") as fh:
        for row in csv.DictReader(fh):
            params[row["simulant"]] = {
                "n": float(row["n"]),
                "k_c": float(row["k_c_kN_per_m_n_plus_1"]),
                "k_phi": float(row["k_phi_kN_per_m_n_plus_2"]),
                "cohesion_kpa": float(row["cohesion_kPa"]),
                "friction_angle_deg": float(row["friction_angle_deg"]),
            }
    return params


def load_experiment_points(
    experiments_csv: Path = DEFAULT_EXPERIMENTS_CSV,
    simulants_csv: Path = DEFAULT_SIMULANTS_CSV,
) -> list[ExperimentPoint]:
    """Read the digitisation worksheet into typed operating points."""
    simulant_params = load_simulant_bekker_params(simulants_csv)
    points: list[ExperimentPoint] = []
    with Path(experiments_csv).open(newline="") as fh:
        for row in csv.DictReader(fh):
            simulant = row["soil_simulant"].strip()
            if simulant not in simulant_params:
                raise KeyError(
                    f"row {row['case_id']!r} references unknown simulant "
                    f"{simulant!r}; not in {simulants_csv}"
                )
            bekker = simulant_params[simulant]
            soil = SoilParameters(
                n=bekker["n"],
                k_c=bekker["k_c"],
                k_phi=bekker["k_phi"],
                cohesion_kpa=bekker["cohesion_kpa"],
                friction_angle_deg=bekker["friction_angle_deg"],
                shear_modulus_k_m=_to_float(row.get("soil_shear_modulus_k_m")),
            )
            wheel = WheelGeometry(
                radius_m=float(row["wheel_radius_m"]),
                width_m=float(row["wheel_width_m"]),
                grouser_height_m=float(row["grouser_height_m"]),
                grouser_count=int(float(row["grouser_count"])),
            )
            points.append(
                ExperimentPoint(
                    source=row["source"].strip(),
                    case_id=row["case_id"].strip(),
                    wheel=wheel,
                    soil=soil,
                    soil_simulant=simulant,
                    vertical_load_n=float(row["vertical_load_n"]),
                    slip=float(row["slip"]),
                    meas_drawbar_pull_n=_to_float(row.get("meas_drawbar_pull_n")),
                    meas_sinkage_m=_to_float(row.get("meas_sinkage_m")),
                    meas_torque_nm=_to_float(row.get("meas_torque_nm")),
                    status=row.get("status", "").strip(),
                    citation=row.get("citation", "").strip(),
                    notes=row.get("notes", "").strip(),
                )
            )
    return points


def _bw_predict(point: ExperimentPoint) -> tuple[float, float, float]:
    """Run the BW kernel; return ``(DP, torque, sinkage)`` or NaNs on failure.

    The kernel raises ``ValueError`` when no entry angle satisfies vertical
    balance (wheel fully buried). We surface that as NaN rather than letting
    one pathological operating point abort the whole sweep.
    """
    try:
        forces = single_wheel_forces(
            point.wheel, point.soil, point.vertical_load_n, point.slip
        )
    except ValueError:
        return math.nan, math.nan, math.nan
    return forces.drawbar_pull_n, forces.driving_torque_nm, forces.sinkage_m


def _abs_pct_error(predicted: float, measured: float) -> float:
    """Absolute percentage error, NaN if measured is missing or zero."""
    if math.isnan(predicted) or math.isnan(measured) or measured == 0.0:
        return math.nan
    return 100.0 * abs(predicted - measured) / abs(measured)


def compare_to_experiment(
    points: list[ExperimentPoint] | None = None,
    *,
    experiments_csv: Path = DEFAULT_EXPERIMENTS_CSV,
    simulants_csv: Path = DEFAULT_SIMULANTS_CSV,
) -> pd.DataFrame:
    """Build a per-point comparison table: measured vs Bekker-Wong.

    Parameters
    ----------
    points
        Pre-loaded operating points; loaded from ``experiments_csv`` if None.

    Returns
    -------
    pandas.DataFrame
        One row per operating point with measured and BW predictions plus
        signed residuals and absolute percentage errors for drawbar pull
        and sinkage.
    """
    if points is None:
        points = load_experiment_points(experiments_csv, simulants_csv)

    records: list[dict[str, object]] = []
    for pt in points:
        bw_dp, bw_tau, bw_z = _bw_predict(pt)

        records.append(
            {
                "source": pt.source,
                "case_id": pt.case_id,
                "soil_simulant": pt.soil_simulant,
                "grouser_height_m": pt.wheel.grouser_height_m,
                "vertical_load_n": pt.vertical_load_n,
                "slip": pt.slip,
                "status": pt.status,
                "meas_drawbar_pull_n": pt.meas_drawbar_pull_n,
                "meas_sinkage_m": pt.meas_sinkage_m,
                "meas_torque_nm": pt.meas_torque_nm,
                "bw_drawbar_pull_n": bw_dp,
                "bw_torque_nm": bw_tau,
                "bw_sinkage_m": bw_z,
                "bw_dp_abs_pct_err": _abs_pct_error(bw_dp, pt.meas_drawbar_pull_n),
                "bw_sinkage_abs_pct_err": _abs_pct_error(bw_z, pt.meas_sinkage_m),
                "citation": pt.citation,
            }
        )

    return pd.DataFrame.from_records(records)


def summarise(df: pd.DataFrame) -> dict[str, float | int]:
    """Aggregate accuracy over the digitised (measured) rows only.

    Returns counts and median absolute percentage errors for the Bekker-Wong
    kernel. Medians over an empty set are NaN.
    """
    digitised = df[df["meas_drawbar_pull_n"].notna()]

    def _median(col: str, frame: pd.DataFrame) -> float:
        values = frame[col].dropna()
        return float(values.median()) if len(values) else math.nan

    return {
        "n_operating_points": int(len(df)),
        "n_digitised": int(len(digitised)),
        "n_pending_digitisation": int((df["status"] == "pending_digitisation").sum()),
        "bw_dp_median_abs_pct_err": _median("bw_dp_abs_pct_err", digitised),
        "bw_sinkage_median_abs_pct_err": _median("bw_sinkage_abs_pct_err", digitised),
    }


__all__ = [
    "DEFAULT_EXPERIMENTS_CSV",
    "DEFAULT_SIMULANTS_CSV",
    "ExperimentPoint",
    "compare_to_experiment",
    "load_experiment_points",
    "load_simulant_bekker_params",
    "summarise",
]
