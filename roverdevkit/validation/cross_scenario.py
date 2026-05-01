"""Cross-scenario ranking + one-at-a-time sensitivity checks.

Two light-weight Week-5 checks layered on top of the evaluator:

1. :func:`rank_archetypes` runs three hand-crafted design archetypes
   (``large_traverser``, ``polar_survivor``, ``slope_climber``) across
   the four canonical scenarios and returns the winner per scenario
   and metric. If the evaluator behaves correctly, each archetype wins
   its own specialty scenario/metric pair.

2. :func:`one_at_a_time_sensitivity` sweeps each design variable one
   at a time around a baseline and reports the sign of the change in
   each :class:`MissionMetrics` field. These signs should match
   physical intuition (e.g. more solar area -> higher energy margin);
   the accompanying tests enforce the expected direction.

Both are a direct response to the "is the evaluator right in the
right *direction* when parameters change?" question from the project
plan §6 W5.
"""

from __future__ import annotations

from dataclasses import dataclass

from roverdevkit.mission.evaluator import evaluate
from roverdevkit.mission.scenarios import load_scenario
from roverdevkit.schema import DesignVector, MissionMetrics, MissionScenario

# ---------------------------------------------------------------------------
# Archetypes
# ---------------------------------------------------------------------------


def _large_traverser() -> DesignVector:
    """Big solar + battery + fast speed; wins on range in benign terrain."""
    return DesignVector(
        wheel_radius_m=0.15,
        wheel_width_m=0.10,
        grouser_height_m=0.008,
        grouser_count=12,
        n_wheels=6,
        chassis_mass_kg=20.0,
        wheelbase_m=0.8,
        solar_area_m2=1.2,
        battery_capacity_wh=300.0,
        avionics_power_w=15.0,
        peak_wheel_torque_nm=2.0,
    )


def _polar_survivor() -> DesignVector:
    """Low-power, small-solar, big-battery: intended to bank night energy."""
    return DesignVector(
        wheel_radius_m=0.10,
        wheel_width_m=0.07,
        grouser_height_m=0.006,
        grouser_count=12,
        n_wheels=4,
        chassis_mass_kg=8.0,
        wheelbase_m=0.5,
        solar_area_m2=0.8,
        battery_capacity_wh=400.0,
        avionics_power_w=10.0,
        peak_wheel_torque_nm=1.0,
    )


def _slope_climber() -> DesignVector:
    """Big grousers, many wheels, heavier duty cycle for pure slope work."""
    return DesignVector(
        wheel_radius_m=0.18,
        wheel_width_m=0.12,
        grouser_height_m=0.012,
        grouser_count=20,
        n_wheels=6,
        chassis_mass_kg=25.0,
        wheelbase_m=1.0,
        solar_area_m2=0.6,
        battery_capacity_wh=200.0,
        avionics_power_w=15.0,
        peak_wheel_torque_nm=4.0,
    )


def archetypes() -> dict[str, DesignVector]:
    """Return the three named archetype design vectors."""
    return {
        "large_traverser": _large_traverser(),
        "polar_survivor": _polar_survivor(),
        "slope_climber": _slope_climber(),
    }


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArchetypeRanking:
    """Who won what under which scoring rule."""

    scenario_name: str
    range_winner: str
    slope_capability_winner: str
    energy_margin_winner: str
    per_archetype: dict[str, MissionMetrics]


def _canonical_scenarios() -> list[MissionScenario]:
    return [
        load_scenario("equatorial_mare_traverse"),
        load_scenario("polar_prospecting"),
        load_scenario("highland_slope_capability"),
        load_scenario("crater_rim_survey"),
    ]


def rank_archetypes() -> dict[str, ArchetypeRanking]:
    """Evaluate every archetype on every canonical scenario.

    Returns a dict keyed by scenario name, with each entry listing the
    winning archetype for three metrics (range, slope capability, energy
    margin) plus the full per-archetype metrics for inspection.
    """
    designs = archetypes()
    out: dict[str, ArchetypeRanking] = {}
    for scenario in _canonical_scenarios():
        per_archetype = {name: evaluate(design, scenario) for name, design in designs.items()}
        range_winner = max(per_archetype, key=lambda n: per_archetype[n].range_km)
        slope_winner = max(per_archetype, key=lambda n: per_archetype[n].slope_capability_deg)
        energy_winner = max(per_archetype, key=lambda n: per_archetype[n].energy_margin_pct)
        out[scenario.name] = ArchetypeRanking(
            scenario_name=scenario.name,
            range_winner=range_winner,
            slope_capability_winner=slope_winner,
            energy_margin_winner=energy_winner,
            per_archetype=per_archetype,
        )
    return out


# ---------------------------------------------------------------------------
# One-at-a-time sensitivity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SensitivityEntry:
    """Effect of bumping one design variable from baseline.

    ``delta_energy_margin_raw_pct`` is the unclipped companion of
    ``delta_energy_margin_pct`` (see
    :class:`roverdevkit.schema.MissionMetrics`) and is what picks up
    changes even when the SOC-based margin has saturated at 0 % or
    100 %. Both are reported so the table shows whether a saturated
    reporting metric is hiding real signal underneath.
    """

    variable: str
    baseline_value: float
    bumped_value: float
    delta_range_km: float
    delta_energy_margin_pct: float
    delta_energy_margin_raw_pct: float
    delta_slope_capability_deg: float
    delta_total_mass_kg: float


def _baseline_design() -> DesignVector:
    """A mid-range design-vector baseline for sensitivity runs.

    Deliberately slow + low-duty so the sensitivity scenario's
    ``traverse_distance_m`` cap is *not* reached within
    ``mission_duration_earth_days``; otherwise the ``delta_range_km``
    values saturate at zero.
    """
    return DesignVector(
        wheel_radius_m=0.12,
        wheel_width_m=0.08,
        grouser_height_m=0.008,
        grouser_count=12,
        n_wheels=6,
        chassis_mass_kg=12.0,
        wheelbase_m=0.6,
        solar_area_m2=0.4,
        battery_capacity_wh=100.0,
        avionics_power_w=15.0,
        peak_wheel_torque_nm=1.5,
    )


def _sensitivity_scenario() -> MissionScenario:
    """Distance-unlimited scenario so range is energy-/duty-bound.

    Uses a synthetic 100 km traverse on equatorial mare so baseline
    designs are never capped by ``traverse_distance_m``; this lets
    `delta_range_km` actually move when we bump solar area or duty
    cycle. Not a canonical scenario - deliberately excluded from
    :func:`list_scenarios`.
    """
    return MissionScenario(
        name="sensitivity_equatorial_long",
        latitude_deg=20.2,
        traverse_distance_m=100_000.0,
        terrain_class="mare_nominal",
        soil_simulant="Apollo_regolith_nominal",
        mission_duration_earth_days=14.0,
        max_slope_deg=8.0,
        sun_geometry="diurnal",
        # Schema v7: drive duty cycle is a single per-scenario parameter
        # (designed_duty_cycle was removed from the design vector). 0.5
        # is generous for an equatorial mare scenario but ensures the
        # baseline design is *not* duty-bound, so the chassis_mass and
        # solar bumps below actually move range_km.
        operational_duty_cycle=0.5,
    )


# Variables we bump, and the +/- direction we apply to stay in-bounds.
# The bump is a small but non-trivial fraction of the range; the sign
# of the resulting metric change is what matters, not the magnitude.
_SENSITIVITY_BUMPS: tuple[tuple[str, float], ...] = (
    ("solar_area_m2", +0.3),
    ("battery_capacity_wh", +100.0),
    ("avionics_power_w", +10.0),
    ("chassis_mass_kg", +8.0),
    ("wheel_radius_m", +0.04),
    ("peak_wheel_torque_nm", +1.5),
)


def one_at_a_time_sensitivity(
    scenario: MissionScenario | None = None,
) -> list[SensitivityEntry]:
    """Sweep each design variable in turn; record the metric deltas.

    Parameters
    ----------
    scenario
        Optional scenario to evaluate against. Defaults to a distance-
        unlimited scenario (:func:`_sensitivity_scenario`) where range
        is energy- and duty-cycle-bound rather than capped by
        ``traverse_distance_m``.
    """
    scenario = scenario or _sensitivity_scenario()
    base = _baseline_design()
    base_metrics = evaluate(base, scenario)

    entries: list[SensitivityEntry] = []
    for var, delta in _SENSITIVITY_BUMPS:
        baseline_val = float(getattr(base, var))
        bumped_val = baseline_val + delta
        bumped = base.model_copy(update={var: bumped_val})
        bumped_metrics = evaluate(bumped, scenario)
        entries.append(
            SensitivityEntry(
                variable=var,
                baseline_value=baseline_val,
                bumped_value=bumped_val,
                delta_range_km=bumped_metrics.range_km - base_metrics.range_km,
                delta_energy_margin_pct=(
                    bumped_metrics.energy_margin_pct - base_metrics.energy_margin_pct
                ),
                delta_energy_margin_raw_pct=(
                    bumped_metrics.energy_margin_raw_pct - base_metrics.energy_margin_raw_pct
                ),
                delta_slope_capability_deg=(
                    bumped_metrics.slope_capability_deg - base_metrics.slope_capability_deg
                ),
                delta_total_mass_kg=(bumped_metrics.total_mass_kg - base_metrics.total_mass_kg),
            )
        )
    return entries
