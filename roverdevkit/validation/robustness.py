"""Layer-6 robustness sweep.

Perturbs each scenario's soil parameters and the bottom-up
:class:`roverdevkit.mass.parametric_mers.MassModelParams` specific-mass
coefficients by ±20 %, re-runs the cross-scenario archetype set under
each perturbation, and reports continuous-metric drift plus the
qualitative-ranking-stability check the §7 Layer-6 entry of
``project_plan.md`` describes:

    "Perturb soil parameters and ``MassModelParams`` specific-mass
    coefficients by ±20 %, re-run optimization, check if qualitative
    conclusions persist."

The sweep is **archetype-based, not optimisation-based** by design:
re-running NSGA-II under every perturbation costs minutes per
scenario × perturbation × archetype, which would explode the runtime
budget without changing the headline finding (the test is whether
the *qualitative ranking* of archetypes shifts under ±20 % parameter
moves, which is observable directly on the archetype set).

Outputs
-------
- :class:`RobustnessEntry` — per (scenario × archetype × perturbation)
  perturbed metrics + signed deltas vs the baseline.
- :class:`RobustnessSummary` — per perturbation: median absolute relative
  drift in each continuous metric across (scenario × archetype) cells,
  plus a per-scenario flag for whether the archetype ranking
  (range / energy-margin / slope-capability winner) is preserved.
- :func:`format_robustness_report` — markdown writer suitable for
  ``reports/layer6_robustness.md`` and the §8.5 paper paragraph.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Literal

from roverdevkit.mass.parametric_mers import MassModelParams
from roverdevkit.mission.evaluator import evaluate_verbose
from roverdevkit.schema import DesignVector, MissionMetrics, MissionScenario
from roverdevkit.terramechanics.bekker_wong import SoilParameters
from roverdevkit.terramechanics.soils import get_soil_parameters
from roverdevkit.validation.cross_scenario import (
    _canonical_scenarios,
    archetypes,
)

# ---------------------------------------------------------------------------
# Perturbation specification
# ---------------------------------------------------------------------------


PerturbationKind = Literal["soil", "mass"]


@dataclass(frozen=True)
class Perturbation:
    """One ±X % perturbation of the soil or mass-model parameters.

    Parameters with multipliers > 1 increase the named coefficient;
    multipliers < 1 decrease it. ``soil`` multipliers act jointly on
    ``k_c`` and ``k_phi`` (the Bekker pressure-sinkage moduli),
    matching the published practice of treating the two as a coupled
    "soil stiffness" pair (Wong 2008 §2.3; the two enter the
    pressure-sinkage law through the same ``(k_c/b + k_phi)`` group
    so individual residuals are not separately identifiable from
    bevameter data alone). ``mass`` multipliers act on a single named
    field of :class:`MassModelParams`.
    """

    name: str
    kind: PerturbationKind
    multiplier: float
    target_field: str | None = None
    """``MassModelParams`` field name; required when ``kind == 'mass'``."""


_DEFAULT_PERTURBATIONS: tuple[Perturbation, ...] = (
    Perturbation(name="soil_stiffness_minus20", kind="soil", multiplier=0.80),
    Perturbation(name="soil_stiffness_plus20", kind="soil", multiplier=1.20),
    Perturbation(
        name="wheel_areal_density_minus20",
        kind="mass",
        multiplier=0.80,
        target_field="wheel_structural_area_density_kg_per_m2",
    ),
    Perturbation(
        name="wheel_areal_density_plus20",
        kind="mass",
        multiplier=1.20,
        target_field="wheel_structural_area_density_kg_per_m2",
    ),
    Perturbation(
        name="solar_areal_mass_minus20",
        kind="mass",
        multiplier=0.80,
        target_field="solar_specific_area_mass_kg_per_m2",
    ),
    Perturbation(
        name="solar_areal_mass_plus20",
        kind="mass",
        multiplier=1.20,
        target_field="solar_specific_area_mass_kg_per_m2",
    ),
    Perturbation(
        name="battery_specific_energy_minus20",
        kind="mass",
        multiplier=0.80,
        target_field="battery_pack_specific_energy_wh_per_kg",
    ),
    Perturbation(
        name="battery_specific_energy_plus20",
        kind="mass",
        multiplier=1.20,
        target_field="battery_pack_specific_energy_wh_per_kg",
    ),
    Perturbation(
        name="motor_specific_torque_minus20",
        kind="mass",
        multiplier=0.80,
        target_field="motor_specific_torque_kg_per_nm",
    ),
    Perturbation(
        name="motor_specific_torque_plus20",
        kind="mass",
        multiplier=1.20,
        target_field="motor_specific_torque_kg_per_nm",
    ),
)


def default_perturbations() -> tuple[Perturbation, ...]:
    """Read-only access to the default ±20 % perturbation set."""
    return _DEFAULT_PERTURBATIONS


# ---------------------------------------------------------------------------
# Apply perturbations
# ---------------------------------------------------------------------------


def perturb_soil(soil: SoilParameters, multiplier: float) -> SoilParameters:
    """Multiply ``k_c`` and ``k_phi`` by ``multiplier``; other fields unchanged."""
    return dataclasses.replace(
        soil,
        k_c=soil.k_c * multiplier,
        k_phi=soil.k_phi * multiplier,
    )


def perturb_mass_params(
    params: MassModelParams, target_field: str, multiplier: float
) -> MassModelParams:
    """Multiply a single named :class:`MassModelParams` field by ``multiplier``."""
    if not hasattr(params, target_field):
        raise ValueError(f"unknown MassModelParams field: {target_field!r}")
    current = getattr(params, target_field)
    return dataclasses.replace(params, **{target_field: current * multiplier})


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RobustnessEntry:
    """Perturbed metrics for one (scenario, archetype, perturbation) cell.

    ``energy_margin_raw_pct`` is reported instead of the SOC-clipped
    :attr:`MissionMetrics.energy_margin_pct` so the drift table picks up
    signal even when the clipped margin saturates at 0 % or 100 % (the
    case for well-budgeted archetypes on benign scenarios).
    """

    scenario_name: str
    archetype: str
    perturbation: str
    baseline_range_km: float
    perturbed_range_km: float
    baseline_energy_margin_raw_pct: float
    perturbed_energy_margin_raw_pct: float
    baseline_slope_capability_deg: float
    perturbed_slope_capability_deg: float
    baseline_total_mass_kg: float
    perturbed_total_mass_kg: float

    @property
    def delta_range_km(self) -> float:
        return self.perturbed_range_km - self.baseline_range_km

    @property
    def delta_energy_margin_raw_pct(self) -> float:
        return self.perturbed_energy_margin_raw_pct - self.baseline_energy_margin_raw_pct

    @property
    def delta_slope_capability_deg(self) -> float:
        return self.perturbed_slope_capability_deg - self.baseline_slope_capability_deg

    @property
    def delta_total_mass_kg(self) -> float:
        return self.perturbed_total_mass_kg - self.baseline_total_mass_kg


@dataclass(frozen=True)
class RobustnessSummary:
    """Aggregate Layer-6 outcome across all (scenario, archetype) cells.

    ``median_abs_rel_drift_energy_margin`` operates on
    ``energy_margin_raw_pct`` so the metric reports real signal even
    where the SOC-clipped reporting margin saturates.
    """

    perturbation: str
    median_abs_rel_drift_range: float
    median_abs_rel_drift_energy_margin: float
    median_abs_rel_drift_slope: float
    median_abs_rel_drift_mass: float
    ranking_preserved_per_scenario: dict[str, bool]
    """``True`` for a scenario when the (range / energy-margin-raw /
    slope) archetype winners under the perturbation match the baseline
    winners."""

    n_cells: int
    entries: list[RobustnessEntry] = field(default_factory=list)

    @property
    def all_rankings_preserved(self) -> bool:
        return all(self.ranking_preserved_per_scenario.values())


def _eval_under(
    design: DesignVector,
    scenario: MissionScenario,
    *,
    soil: SoilParameters | None,
    mass_params: MassModelParams | None,
) -> MissionMetrics:
    return evaluate_verbose(
        design,
        scenario,
        mass_params=mass_params,
        soil_override=soil,
    ).metrics


def _winners(per_archetype: dict[str, MissionMetrics]) -> tuple[str, str, str]:
    range_w = max(per_archetype, key=lambda n: per_archetype[n].range_km)
    em_w = max(per_archetype, key=lambda n: per_archetype[n].energy_margin_raw_pct)
    slope_w = max(per_archetype, key=lambda n: per_archetype[n].slope_capability_deg)
    return range_w, em_w, slope_w


def _safe_rel(num: float, den: float, eps: float = 1.0) -> float:
    return abs(num) / max(abs(den), eps)


def cross_scenario_robustness_sweep(
    perturbations: tuple[Perturbation, ...] = _DEFAULT_PERTURBATIONS,
    *,
    designs: dict[str, DesignVector] | None = None,
    scenarios: list[MissionScenario] | None = None,
) -> list[RobustnessSummary]:
    """Run the Layer-6 archetype × scenario × perturbation sweep.

    Parameters
    ----------
    perturbations
        Iterable of :class:`Perturbation`. Defaults to the canonical
        ±20 % set returned by :func:`default_perturbations`.
    designs
        Mapping ``archetype_name -> DesignVector``. Defaults to the
        three :func:`roverdevkit.validation.cross_scenario.archetypes`
        archetypes.
    scenarios
        Scenarios to evaluate over. Defaults to the four canonical
        scenarios (equatorial mare traverse, polar prospecting,
        highland slope capability, crater rim survey).

    Returns
    -------
    list[RobustnessSummary]
        One :class:`RobustnessSummary` per perturbation, in the order
        of the input ``perturbations`` tuple. Each summary aggregates
        every (scenario × archetype) cell under that perturbation.
    """
    designs = designs or archetypes()
    scenarios = scenarios or _canonical_scenarios()

    baseline_per_scenario: dict[str, dict[str, MissionMetrics]] = {}
    baseline_winners_per_scenario: dict[str, tuple[str, str, str]] = {}
    for scenario in scenarios:
        scenario_baseline: dict[str, MissionMetrics] = {}
        for arche_name, design in designs.items():
            scenario_baseline[arche_name] = _eval_under(
                design,
                scenario,
                soil=None,
                mass_params=None,
            )
        baseline_per_scenario[scenario.name] = scenario_baseline
        baseline_winners_per_scenario[scenario.name] = _winners(scenario_baseline)

    summaries: list[RobustnessSummary] = []
    for perturbation in perturbations:
        rel_drift_range: list[float] = []
        rel_drift_em: list[float] = []
        rel_drift_slope: list[float] = []
        rel_drift_mass: list[float] = []
        ranking_preserved: dict[str, bool] = {}
        entries: list[RobustnessEntry] = []

        for scenario in scenarios:
            base_simulant_soil = get_soil_parameters(scenario.soil_simulant)
            soil_override: SoilParameters | None = None
            mass_override: MassModelParams | None = None
            if perturbation.kind == "soil":
                soil_override = perturb_soil(base_simulant_soil, perturbation.multiplier)
            elif perturbation.kind == "mass":
                if perturbation.target_field is None:
                    raise ValueError(f"mass perturbation {perturbation.name} missing target_field")
                mass_override = perturb_mass_params(
                    MassModelParams(),
                    perturbation.target_field,
                    perturbation.multiplier,
                )

            scenario_perturbed: dict[str, MissionMetrics] = {}
            for arche_name, design in designs.items():
                base = baseline_per_scenario[scenario.name][arche_name]
                pert = _eval_under(
                    design,
                    scenario,
                    soil=soil_override,
                    mass_params=mass_override,
                )
                scenario_perturbed[arche_name] = pert
                entries.append(
                    RobustnessEntry(
                        scenario_name=scenario.name,
                        archetype=arche_name,
                        perturbation=perturbation.name,
                        baseline_range_km=base.range_km,
                        perturbed_range_km=pert.range_km,
                        baseline_energy_margin_raw_pct=base.energy_margin_raw_pct,
                        perturbed_energy_margin_raw_pct=pert.energy_margin_raw_pct,
                        baseline_slope_capability_deg=base.slope_capability_deg,
                        perturbed_slope_capability_deg=pert.slope_capability_deg,
                        baseline_total_mass_kg=base.total_mass_kg,
                        perturbed_total_mass_kg=pert.total_mass_kg,
                    )
                )
                rel_drift_range.append(_safe_rel(pert.range_km - base.range_km, base.range_km, 0.1))
                rel_drift_em.append(
                    _safe_rel(
                        pert.energy_margin_raw_pct - base.energy_margin_raw_pct,
                        base.energy_margin_raw_pct,
                        1.0,
                    )
                )
                rel_drift_slope.append(
                    _safe_rel(
                        pert.slope_capability_deg - base.slope_capability_deg,
                        base.slope_capability_deg,
                        1.0,
                    )
                )
                rel_drift_mass.append(
                    _safe_rel(
                        pert.total_mass_kg - base.total_mass_kg,
                        base.total_mass_kg,
                        0.1,
                    )
                )

            ranking_preserved[scenario.name] = _winners(scenario_perturbed) == (
                baseline_winners_per_scenario[scenario.name]
            )

        def _median(xs: list[float]) -> float:
            sorted_xs = sorted(xs)
            n = len(sorted_xs)
            if n == 0:
                return 0.0
            if n % 2 == 1:
                return sorted_xs[n // 2]
            return 0.5 * (sorted_xs[n // 2 - 1] + sorted_xs[n // 2])

        summaries.append(
            RobustnessSummary(
                perturbation=perturbation.name,
                median_abs_rel_drift_range=_median(rel_drift_range),
                median_abs_rel_drift_energy_margin=_median(rel_drift_em),
                median_abs_rel_drift_slope=_median(rel_drift_slope),
                median_abs_rel_drift_mass=_median(rel_drift_mass),
                ranking_preserved_per_scenario=ranking_preserved,
                n_cells=len(rel_drift_range),
                entries=entries,
            )
        )
    return summaries


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt_pct(x: float) -> str:
    return f"{100.0 * x:6.1f} %"


def format_robustness_report(summaries: list[RobustnessSummary]) -> str:
    """Render a :func:`cross_scenario_robustness_sweep` result as markdown."""
    lines: list[str] = []
    lines.append("# Layer-6 robustness sweep")
    lines.append("")
    lines.append(
        "Each perturbation is applied to every canonical scenario × archetype "
        "cell. Continuous-metric drift is reported as the median absolute "
        "relative change vs the unperturbed baseline. Ranking preservation "
        "asks whether the (range / energy-margin / slope-capability) "
        "archetype winners change under the perturbation."
    )
    lines.append("")
    lines.append(
        "Method: archetype-based rather than NSGA-II-rerun. The archetype "
        "set is small enough to test the qualitative-ranking-stability "
        "claim directly without re-running multi-objective optimization "
        "under every perturbation."
    )
    lines.append("")
    lines.append("## Continuous-metric drift")
    lines.append("")
    lines.append(
        "Energy margin uses the unclipped ``energy_margin_raw_pct`` "
        "(see :class:`MissionMetrics`) so the column does not silently "
        "saturate at 0 % / 100 % on energy-rich archetypes."
    )
    lines.append("")
    lines.append(
        "| Perturbation | range_km | energy_margin_raw_pct | slope_deg | total_mass_kg | n cells |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for s in summaries:
        lines.append(
            "| `{name}` | {r} | {em} | {sl} | {m} | {n} |".format(
                name=s.perturbation,
                r=_fmt_pct(s.median_abs_rel_drift_range),
                em=_fmt_pct(s.median_abs_rel_drift_energy_margin),
                sl=_fmt_pct(s.median_abs_rel_drift_slope),
                m=_fmt_pct(s.median_abs_rel_drift_mass),
                n=s.n_cells,
            )
        )
    lines.append("")
    lines.append("## Ranking preservation per scenario")
    lines.append("")
    scenario_names = sorted({sn for s in summaries for sn in s.ranking_preserved_per_scenario})
    header = ["Perturbation", *scenario_names, "All preserved?"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for s in summaries:
        cells: list[str] = [f"`{s.perturbation}`"]
        for name in scenario_names:
            ok = s.ranking_preserved_per_scenario.get(name, False)
            cells.append("yes" if ok else "**no**")
        cells.append("yes" if s.all_rankings_preserved else "**no**")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        "Reading: a `yes` row means every archetype winner is preserved "
        "under that perturbation. A `no` cell means at least one of the "
        "three winners (range / energy-margin / slope) flipped, which is "
        "the operational definition of 'qualitative conclusions did not "
        "persist' from §7 Layer 6 of the project plan."
    )
    return "\n".join(lines) + "\n"


__all__ = [
    "Perturbation",
    "RobustnessEntry",
    "RobustnessSummary",
    "cross_scenario_robustness_sweep",
    "default_perturbations",
    "format_robustness_report",
    "perturb_mass_params",
    "perturb_soil",
]
