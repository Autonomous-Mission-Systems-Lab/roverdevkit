"""Run the mission evaluator on published rovers and score vs ground truth.

This module is the real-rover validation analogue of :mod:`roverdevkit.mass.validation`:
it exposes a tidy :class:`RoverComparisonResult` for each rover in the
:mod:`roverdevkit.validation.rover_registry`, a
:class:`ComparisonSummary` that aggregates across the set, and a
CI-enforceable acceptance gate via :func:`acceptance_gate`.

What "validation" means here
---------------------------------------------------------------

The evaluator produces a *design-space upper bound* on range: given
``delta >= 0.1`` (the schema's drive-duty floor) and constant-speed
driving, the predicted traverse is larger than what real rover ops
typically achieve, because real missions interleave long science and
thermal-wait windows that the model does not capture.

So the acceptance criteria are explicitly:

1. **Range feasibility.** Predicted >= published floor. Missing this
   means the rover *cannot* reach the distance it actually flew; a
   much stronger failure signal than a simple ratio test.
2. **Range sanity ceiling.** Predicted <= 10x published ceiling.
   Catches pathological over-prediction (e.g. evaluator ignoring a
   broken motor or a zeroed avionics load).
3. **Thermal survival match.** Sim's hot+cold steady-state prediction
   matches the published outcome exactly. real-rover validation's clearest binary
   check.
4. **No stall / motor overload.** ``stalled`` must be False (schema v6),
   reflecting that the rover *can* move at all in the scenario soil +
   slope and that the slip-balance torque demand stays inside the
   designed per-wheel torque envelope.
5. **Peak solar power in-band.** Predicted within the published
   low/high band, where the band already accounts for dust,
   temperature derating, and seasonal irradiance.

Aggregate reporting follows the mass-validation mass-validation pattern: a formatted
human-readable report and a tidy structure that a CI test consumes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from roverdevkit.mission.evaluator import evaluate
from roverdevkit.power.solar import (
    SOLAR_CONSTANT_AU_1_W_PER_M2,
    panel_power_w,
    sun_elevation_deg,
)
from roverdevkit.schema import MissionMetrics
from roverdevkit.validation.rover_registry import (
    PublishedTruth,
    RoverRegistryEntry,
    flown_registry,
    load_truth_table,
    truth_by_rover,
)

# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoverComparisonResult:
    """Evaluator output alongside published truth for one rover."""

    rover_name: str
    metrics: MissionMetrics
    truth: PublishedTruth
    peak_solar_power_w_predicted: float

    # Per-criterion booleans for transparency.
    range_feasible: bool
    range_below_sanity_ceiling: bool
    thermal_matches: bool
    motor_and_traversal_ok: bool
    peak_solar_in_band: bool

    @property
    def passes(self) -> bool:
        """True iff every acceptance criterion fires."""
        return (
            self.range_feasible
            and self.range_below_sanity_ceiling
            and self.thermal_matches
            and self.motor_and_traversal_ok
            and self.peak_solar_in_band
        )

    @property
    def range_m_predicted(self) -> float:
        return self.metrics.range_km * 1000.0

    @property
    def range_ratio(self) -> float:
        """Predicted / published traverse distance."""
        return self.range_m_predicted / max(1e-9, self.truth.traverse_m_published)

    @property
    def peak_solar_ratio(self) -> float:
        return self.peak_solar_power_w_predicted / max(
            1e-9, self.truth.peak_solar_power_w_published
        )


@dataclass(frozen=True)
class ComparisonSummary:
    """Per-rover results plus an overall pass/fail aggregate."""

    results: tuple[RoverComparisonResult, ...]
    n_pass: int
    n_total: int

    @property
    def all_pass(self) -> bool:
        return self.n_pass == self.n_total


# ---------------------------------------------------------------------------
# Peak-solar prediction (registry-aware)
# ---------------------------------------------------------------------------


def _predicted_peak_solar_power_w(entry: RoverRegistryEntry) -> float:
    """Closed-form peak noon power for this rover's design + scenario.

    Computed independently of the traverse loop so we can compare to
    published peak-power values without having to dig it out of the
    time-history arrays. Uses the same panel-physics model the traverse
    sim does (:func:`panel_power_w`) at peak sun elevation, with the
    panel orientation (``panel_tilt_deg``, ``panel_azimuth_deg``)
    taken from the registry entry. Flown rovers (Pragyan, Yutu-2)
    keep ``panel_tilt_deg = 0`` because their published peak-solar
    bands in ``data/published_traverse_data.csv`` are operational
    averages calibrated against horizontal-equivalent pointing;
    polar design-target rovers (MoonRanger, CADRE-unit) carry
    deployable tilted arrays and have non-zero values in the
    registry.

    Lunar-only since the Mars-gravity Sojourner sentinel was removed
    The 1 AU solar constant is the right
    value for every current registry entry.
    """
    peak_elev = sun_elevation_deg(entry.scenario.latitude_deg, lunar_hour_angle_deg=0.0)
    if peak_elev <= 0.0:
        return 0.0

    # Sun azimuth at noon: 0 (north) for southern-hemisphere rovers,
    # 180 (south) for northern. The exact sign convention from
    # `solar.sun_azimuth_deg` is singular at H=0 so we set it
    # explicitly here.
    sun_az_noon = 0.0 if entry.scenario.latitude_deg < 0.0 else 180.0
    return panel_power_w(
        panel_area_m2=entry.design.solar_area_m2,
        panel_efficiency=entry.panel_efficiency,
        sun_elevation_deg=peak_elev,
        panel_tilt_deg=entry.panel_tilt_deg,
        panel_azimuth_deg=entry.panel_azimuth_deg,
        sun_azimuth_deg=sun_az_noon,
        dust_degradation_factor=entry.panel_dust_factor,
        solar_constant_w_per_m2=SOLAR_CONSTANT_AU_1_W_PER_M2,
    )


# ---------------------------------------------------------------------------
# Single-rover scoring
# ---------------------------------------------------------------------------


def compare_one(
    entry: RoverRegistryEntry,
    *,
    truth: PublishedTruth | None = None,
    range_sanity_ceiling_multiple: float = 10.0,
) -> RoverComparisonResult:
    """Run the evaluator on one registry entry and score vs truth.

    Parameters
    ----------
    entry
        Rover + scenario + gravity + thermal architecture triple.
    truth
        Optional explicit :class:`PublishedTruth` override. Defaults to
        :func:`truth_by_rover` so the caller needn't plumb the CSV.
    range_sanity_ceiling_multiple
        Predicted range must not exceed this multiple of the published
        ceiling. 10x is a loose sanity check; structural over-predicts
        (factor ~5x) are expected per docstring above.
    """
    if truth is None:
        truth = truth_by_rover(entry.rover_name)

    metrics = evaluate(
        entry.design,
        entry.scenario,
        gravity_m_per_s2=entry.gravity_m_per_s2,
        thermal_architecture=entry.thermal_architecture,
        panel_tilt_deg=entry.panel_tilt_deg,
        panel_azimuth_deg=entry.panel_azimuth_deg,
    )
    peak_solar_predicted = _predicted_peak_solar_power_w(entry)
    range_m = metrics.range_km * 1000.0

    range_feasible = range_m >= truth.traverse_m_low
    range_below_sanity_ceiling = range_m <= range_sanity_ceiling_multiple * truth.traverse_m_high
    thermal_matches = metrics.thermal_survival == truth.thermal_survival_published
    motor_and_traversal_ok = not bool(metrics.stalled)
    peak_solar_in_band = (
        truth.peak_solar_power_w_low <= peak_solar_predicted <= truth.peak_solar_power_w_high
    )

    return RoverComparisonResult(
        rover_name=entry.rover_name,
        metrics=metrics,
        truth=truth,
        peak_solar_power_w_predicted=peak_solar_predicted,
        range_feasible=range_feasible,
        range_below_sanity_ceiling=range_below_sanity_ceiling,
        thermal_matches=thermal_matches,
        motor_and_traversal_ok=motor_and_traversal_ok,
        peak_solar_in_band=peak_solar_in_band,
    )


# ---------------------------------------------------------------------------
# Full-set scoring
# ---------------------------------------------------------------------------


def compare_all(
    *,
    csv_path: Path | str | None = None,
    range_sanity_ceiling_multiple: float = 10.0,
) -> ComparisonSummary:
    """Run the evaluator on every flown rover in the registry and aggregate.

    Iterates :func:`flown_registry` rather than :func:`registry` because
    Layer-0 truth comparison only makes sense for rovers with actual
    published flight data; design-target entries (MoonRanger, Rashid-1)
    are skipped here and only participate in the baseline-surrogate Layer-1
    surrogate sanity check.

    Parameters
    ----------
    csv_path
        Optional override for ``data/published_traverse_data.csv``.
    range_sanity_ceiling_multiple
        Forwarded to :func:`compare_one`.
    """
    truths = {row.rover_name: row for row in load_truth_table(csv_path)}
    results = tuple(
        compare_one(
            entry,
            truth=truths[entry.rover_name],
            range_sanity_ceiling_multiple=range_sanity_ceiling_multiple,
        )
        for entry in flown_registry()
    )
    n_pass = sum(1 for r in results if r.passes)
    return ComparisonSummary(results=results, n_pass=n_pass, n_total=len(results))


# ---------------------------------------------------------------------------
# CI acceptance gate
# ---------------------------------------------------------------------------


def acceptance_gate(summary: ComparisonSummary) -> None:
    """Raise AssertionError if any rover fails any acceptance criterion.

    Used by :file:`tests/test_rover_comparison.py`. Error message lists
    every failing criterion per rover so the failing CI output is
    self-diagnosing.
    """
    failures: list[str] = []
    for r in summary.results:
        reasons: list[str] = []
        if not r.range_feasible:
            reasons.append(
                f"range infeasible: predicted {r.range_m_predicted:.0f} m "
                f"< published floor {r.truth.traverse_m_low:.0f} m"
            )
        if not r.range_below_sanity_ceiling:
            reasons.append(
                f"range above sanity ceiling: predicted "
                f"{r.range_m_predicted:.0f} m > 10x published "
                f"{r.truth.traverse_m_high:.0f} m"
            )
        if not r.thermal_matches:
            reasons.append(
                f"thermal mismatch: predicted {r.metrics.thermal_survival}, "
                f"published {r.truth.thermal_survival_published}"
            )
        if not r.motor_and_traversal_ok:
            reasons.append("rover stalled (schema v6 stall gate)")
        if not r.peak_solar_in_band:
            reasons.append(
                f"peak solar out of band: predicted "
                f"{r.peak_solar_power_w_predicted:.1f} W, band "
                f"{r.truth.peak_solar_power_w_low:.0f}-"
                f"{r.truth.peak_solar_power_w_high:.0f} W"
            )
        if reasons:
            failures.append(f"{r.rover_name}: " + "; ".join(reasons))
    if failures:
        raise AssertionError(
            "real-rover validation gate failed:\n  - " + "\n  - ".join(failures)
        )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_report(summary: ComparisonSummary) -> str:
    """Human-readable table for notebooks and reports."""
    lines = [
        "Rover      range_pred   range_pub  range_ratio  peak_solar_pred  peak_solar_pub  thermal  motor  PASS?",
        "-" * 105,
    ]
    for r in summary.results:
        lines.append(
            f"{r.rover_name:10s} {r.range_m_predicted:8.0f} m  "
            f"{r.truth.traverse_m_published:8.0f} m  "
            f"{r.range_ratio:9.2f}x  "
            f"{r.peak_solar_power_w_predicted:12.1f} W  "
            f"{r.truth.peak_solar_power_w_published:12.1f} W  "
            f"{str(r.metrics.thermal_survival) == str(r.truth.thermal_survival_published) and 'match' or 'MISS':5s}   "
            f"{'ok' if r.motor_and_traversal_ok else 'STALL':5s}  "
            f"{'PASS' if r.passes else 'FAIL'}"
        )
    lines.append("-" * 105)
    lines.append(f"Pass rate: {summary.n_pass}/{summary.n_total}")
    return "\n".join(lines)
