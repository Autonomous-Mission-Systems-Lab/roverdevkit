"""Corrected mission evaluator dispatch for ``POST /evaluate``.

This service is a thin wrapper around
:func:`roverdevkit.mission.evaluator.evaluate_verbose`. The single-design
panel calls it for the deterministic median of each performance metric so
the chart's diamond marker is the ground-truth physics output rather
than the surrogate's regression of it; the surrogate's quantile heads
still supply the prediction interval around that median.

We use ``evaluate_verbose`` (rather than the lighter ``evaluate``) so we
can surface the *why* behind the constraint flags: the peak / cold
enclosure temperatures from the lumped-parameter thermal model and the
explicit drivetrain stall gate (peak per-wheel hub torque demand vs
``DesignVector.peak_wheel_torque_nm``). The cost of the verbose path is
identical -- the underlying physics call is the same -- and the extra
fields are dropped on the floor for callers that only want
``MissionMetrics``.

Schema v6 (W12 step B): the previous ``MotorTorqueDiagnostic`` was
replaced by :class:`StallDiagnostic`. The pre-v6 diagnostic compared the
peak observed torque to a closed-form per-wheel ceiling derived from
``mass × g / N × R × sf × μ`` inside the mass model; v6 makes the
ceiling an explicit design input (``peak_wheel_torque_nm``) and the
stall gate is an explicit slip-balance comparison inside
:mod:`roverdevkit.drivetrain.motor`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from roverdevkit.mission.evaluator import evaluate_verbose
from roverdevkit.power.thermal import ThermalResult
from roverdevkit.schema import DesignVector, MissionMetrics, MissionScenario
from roverdevkit.surrogate.features import PRIMARY_REGRESSION_TARGETS
from roverdevkit.terramechanics.correction_model import WheelLevelCorrection


@dataclass(frozen=True)
class StallDiagnostic:
    """Drivetrain stall status (schema v6).

    Encodes the explicit stall gate:
    ``stalled = (T_req_per_wheel_nm > peak_wheel_torque_nm) or
    slip_solver_failed``. ``peak_torque_demand_nm`` is the slip-balance
    torque the wheel-level solve computed; ``peak_torque_capacity_nm``
    is the design input (``DesignVector.peak_wheel_torque_nm``) echoed
    back so the frontend can render both numbers side-by-side.
    """

    stalled: bool
    """``True`` iff the rover stalled under the scenario's worst-case
    load (drives :data:`MissionMetrics.stalled`)."""

    peak_torque_demand_nm: float
    """Largest absolute per-wheel torque the slip-balance solve
    demanded during the traverse."""

    peak_torque_capacity_nm: float
    """Design-input drivetrain capacity
    (``DesignVector.peak_wheel_torque_nm``)."""


@dataclass(frozen=True)
class EvaluatorOutput:
    """Container the evaluate route translates into the HTTP response.

    Splitting this off from the Pydantic ``EvaluateResponse`` keeps the
    service layer dependency-free (it only knows core types) and makes
    the route a one-liner.
    """

    metrics: MissionMetrics
    thermal: ThermalResult
    stall: StallDiagnostic
    effective_duty_cycle: float
    cruise_speed_mps: float
    elapsed_ms: float
    used_scm_correction: bool


def evaluate_design(
    design: DesignVector,
    scenario: MissionScenario,
    *,
    correction: WheelLevelCorrection | None,
    operational_duty_cycle: float | None = None,
) -> EvaluatorOutput:
    """Run the corrected mission evaluator on one design × one scenario.

    Parameters
    ----------
    design
        Validated 11-D design vector (Pydantic has already enforced the
        bounds at the HTTP boundary).
    scenario
        One of the canonical scenarios resolved server-side.
    correction
        The shared wheel-level SCM correction artifact (loaded once per
        process by :func:`webapp.backend.loaders.get_correction`). Pass
        ``None`` to fall back to the BW-only evaluator; the route
        decides whether that fallback is acceptable.
    operational_duty_cycle
        Schema v6 (W12 step B): per-call override of
        ``MissionScenario.operational_duty_cycle``. ``None`` (default)
        uses the scenario's calibrated value. Schema v7 (W12 step B
        follow-up): used directly as ``δ_eff`` (clamped to ``[0, 1]``).

    Returns
    -------
    EvaluatorOutput
        :class:`MissionMetrics` plus a wall-clock measurement, an SCM-
        correction flag, the :class:`ThermalResult`, the
        :class:`StallDiagnostic`, and the runtime-resolved
        ``effective_duty_cycle`` / ``cruise_speed_mps``.
    """
    t0 = time.perf_counter()
    detailed = evaluate_verbose(
        design,
        scenario,
        use_scm_correction=correction is not None,
        correction=correction,
        operational_duty_cycle=operational_duty_cycle,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    stall = StallDiagnostic(
        stalled=bool(detailed.metrics.stalled),
        peak_torque_demand_nm=float(detailed.log.peak_torque_demand_nm),
        peak_torque_capacity_nm=float(detailed.log.peak_torque_capacity_nm),
    )

    return EvaluatorOutput(
        metrics=detailed.metrics,
        thermal=detailed.thermal,
        stall=stall,
        effective_duty_cycle=float(detailed.log.effective_duty_cycle),
        cruise_speed_mps=float(detailed.log.cruise_speed_mps),
        elapsed_ms=elapsed_ms,
        used_scm_correction=correction is not None,
    )


def metrics_as_primary_dict(metrics: MissionMetrics) -> dict[str, float]:
    """Project ``MissionMetrics`` onto the four primary regression targets.

    The primary subset is what the surrogate predicts and what the
    chart renders, so the projection lives next to the dispatch to
    keep the column ordering aligned with
    :data:`roverdevkit.surrogate.features.PRIMARY_REGRESSION_TARGETS`.
    """
    src = {
        "range_km": metrics.range_km,
        "energy_margin_raw_pct": metrics.energy_margin_raw_pct,
        "slope_capability_deg": metrics.slope_capability_deg,
        "total_mass_kg": metrics.total_mass_kg,
    }
    return {target: float(src[target]) for target in PRIMARY_REGRESSION_TARGETS}
