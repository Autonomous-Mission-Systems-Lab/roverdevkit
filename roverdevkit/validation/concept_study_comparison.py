"""Layer-6 concept-study head-to-head comparison.

Distinct from Layer-5 rediscovery in framing and constraint set:

- **Rediscovery (Layer-5)** asks "given the rover's actual mass
  budget, does the optimiser's Pareto front sit close to the real
  rover's design vector?" The mass-budget constraint is the rover's
  own modelled mass × (1 + slop), so the optimiser is forced to
  search the slice of the design cube that the rover itself
  occupies. The output measures rediscovery proximity.
- **Concept-study comparison (Layer-6, this module)** asks "if a
  pre-Phase-A design team had this tool, what would the Pareto
  front have told them about their concept's published design
  point?" The mass-budget constraint is the **class envelope**
  (the schema ceiling, currently 50 kg total) rather than the
  concept's own mass — pre-Phase-A by definition does not yet
  have a fixed mass budget. The output measures Pareto-dominance
  (how many alternative designs strictly improve all objectives)
  and the per-objective best-versus-published deltas.

Both layers share the same NSGA-II machinery (``NSGA2Runner``) and
the same leakage controls (class-generic ``*_micro`` scenarios,
``δ_ops = 0.10`` class-neutral anchor) — they differ only in the
constraint set used and the comparison the output reports.

Concept-study entries live in :data:`CONCEPT_STUDIES`. Each entry
carries:

- the cited design vector (``published_design``),
- the cited concept-study scenario (a class-generic ``*_micro``
  YAML so the leakage controls remain consistent across all
  comparisons),
- a ``citation`` string for the writeup,
- a ``notes`` field documenting any back-solves / imputations the
  paper required and any caveats the comparison should disclose,
- and optionally a ``published_metrics`` dict if the paper
  reported headline numbers (range, mass) directly. The
  comparator predicts metrics for the published design under the
  analytical evaluator and reports both numbers side-by-side so a
  reviewer can see where our prediction differs from the paper's.

The first concept-study entry is MoonRanger from Kumar et al.
i-SAIRAS 2020 #5068. VIPER is a deliberate stretch goal once a
publicly cited Phase-A design vector is consolidated.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from roverdevkit.mission.evaluator import evaluate as evaluator_evaluate
from roverdevkit.mission.scenarios import load_scenario
from roverdevkit.schema import DesignVector, MissionScenario
from roverdevkit.surrogate.uncertainty import QuantileHeads
from roverdevkit.terramechanics.soils import get_soil_parameters
from roverdevkit.tradespace.optimizer import (
    DEFAULT_OBJECTIVES,
    NSGA2Runner,
    OptimizationBackend,
    OptimizationConstraint,
    OptimizationObjective,
    OptimizationResult,
)
from roverdevkit.validation.rover_rediscovery import (
    _CONTINUOUS_VARIABLES,
    _dominates,
    _normalised_l2,
)


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConceptStudyEntry:
    """One published pre-Phase-A concept study with cited design + scenario.

    Attributes
    ----------
    rover_name
        Short key used by the CLI and the report file names.
    published_design
        The design vector as published in the concept paper. Carries
        any back-solved fields documented in ``notes``.
    scenario_name
        Which class-generic ``*_micro`` scenario to run the
        comparison under. We deliberately use a class-generic
        scenario (not a per-concept YAML) so the leakage controls
        documented in :mod:`roverdevkit.validation.rover_rediscovery`
        apply to concept-study comparisons too.
    citation
        Single-string canonical citation suitable for a paper
        reference list (author, venue, year, identifier).
    notes
        Free-form prose describing the imputations / back-solves
        the concept paper required and any reviewer caveats that
        belong in the writeup.
    published_metrics
        Optional ``{target -> value}`` dict of numbers the paper
        reports directly (range, total mass, slope capability,
        cruise speed). Used for a paper-vs-our-evaluator
        side-by-side; missing keys are simply omitted from the
        comparison table.
    mass_envelope_kg
        Pre-Phase-A class envelope for the mass-budget constraint.
        Defaults to ``None`` (no constraint, full-class search).
        Concept studies that explicitly state a lander payload
        limit (e.g. "fit inside the CLPS small-rover envelope")
        can pin this to a hard ceiling.
    """

    rover_name: str
    published_design: DesignVector
    scenario_name: str
    citation: str
    notes: str
    published_metrics: dict[str, float] = field(default_factory=dict)
    mass_envelope_kg: float | None = None

    def load_scenario(self) -> MissionScenario:
        return load_scenario(self.scenario_name)


@dataclass(frozen=True)
class ConceptStudyComparison:
    """Result of one head-to-head between a published concept and a Pareto front.

    Attributes
    ----------
    rover_name
        From the :class:`ConceptStudyEntry`.
    scenario_name
        Class-generic ``*_micro`` scenario used for the run.
    backend_used
        ``"evaluator"`` or ``"surrogate"`` (for the optimiser; the
        head-to-head metrics for the published design are always
        computed by the analytical evaluator so the apples-to-apples
        objective comparison is well-defined regardless of which
        backend NSGA-II used).
    published_metrics_predicted
        ``{target -> value}`` from running the analytical evaluator
        on ``published_design`` under ``scenario_name``.
    published_metrics_cited
        ``{target -> value}`` reported directly in the concept paper
        (subset of ``published_metrics_predicted``'s keys).
    pareto_designs, pareto_metrics
        The optimiser's recommended Pareto front under the concept
        study's scenario and mass envelope.
    n_pareto_points_dominating_concept
        How many Pareto points strictly dominate the published
        design on every objective. Headline acceptance signal:
        > 0 means "the optimiser found a strictly better
        design under the same physics".
    n_pareto_points_dominated_by_concept
        Count of Pareto points the published design itself
        dominates. By construction of NSGA-II this is always
        zero on a well-converged run; we record it as a sanity
        check.
    nearest_pareto_index
        Argmin over the front of the normalised-L2 design-space
        distance to ``published_design``.
    nearest_pareto_design_space_distance
        That distance. Same metric used by Layer-5 rediscovery; a
        value of 1.0 is the uniform-random-pair baseline.
    best_per_objective
        ``{target -> {pareto_index, value, concept_value, delta,
        percent_delta}}`` for each objective. The "best" Pareto
        point per objective is the one with the most favourable
        value in the objective's direction (max for ``range_km``
        and ``slope_capability_deg``, min for ``total_mass_kg``).
    optimization_result
        Full :class:`OptimizationResult` so downstream consumers
        can inspect checkpoints / use the Pareto front directly.
    """

    rover_name: str
    scenario_name: str
    backend_used: OptimizationBackend
    published_design: DesignVector
    published_metrics_predicted: dict[str, float]
    published_metrics_cited: dict[str, float]
    pareto_designs: list[DesignVector]
    pareto_metrics: list[dict[str, float]]
    n_pareto_points_dominating_concept: int
    n_pareto_points_dominated_by_concept: int
    nearest_pareto_index: int
    nearest_pareto_design_space_distance: float
    best_per_objective: dict[str, dict[str, float]]
    optimization_result: OptimizationResult


# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------


# Cited specs from Kumar et al. i-SAIRAS 2020 #5068; matches the
# MoonRanger entry in :mod:`roverdevkit.validation.rover_registry`
# (single source of truth for the design vector to avoid drift).
# We re-state the design here so the concept-study entry is self-
# contained for the artifact writer.
#
# Note on ``chassis_mass_kg``: the schema's ``chassis_mass_kg`` is the
# structural-chassis-only mass (the bottom-up parametric model adds
# wheels, motors, solar, battery, avionics, harness, thermal, and a
# 25 % growth margin). Kumar 2020's published "13 kg" is the full-up
# rover mass; we back-solve chassis = 4.5 (≈ 35 % of total) to match
# that headline, consistent with ``data/mass_validation_set.csv``
# and the rest of the registry. See the ``rover_registry`` module
# docstring for the registry-wide audit.
_MOONRANGER_KUMAR_2020_DESIGN = DesignVector(
    wheel_radius_m=0.10,
    wheel_width_m=0.08,
    grouser_height_m=0.012,
    grouser_count=12,
    n_wheels=4,
    chassis_mass_kg=4.5,  # back-solved from published 13 kg full-up
    wheelbase_m=0.40,
    solar_area_m2=0.30,
    battery_capacity_wh=100.0,
    avionics_power_w=25.0,
    peak_wheel_torque_nm=0.75,
)

CONCEPT_STUDIES: dict[str, ConceptStudyEntry] = {
    "MoonRanger": ConceptStudyEntry(
        rover_name="MoonRanger",
        published_design=_MOONRANGER_KUMAR_2020_DESIGN,
        # Scenario choice: at Layer 6 we use the rover's own canonical
        # scenario (``moonranger_polar_demo``) rather than the
        # class-generic ``polar_micro``. The Layer-5 leakage controls
        # do NOT apply here because Layer 6 measures dominance and
        # objective deltas (not proximity to the real design vector);
        # the per-rover canonical scenario is the concept team's
        # stated mission requirement, not leaked ops history. See
        # the module docstring for the framing distinction.
        scenario_name="moonranger_polar_demo",
        citation=(
            "Kumar et al., 'MoonRanger: A Polar Micro-Rover Mission "
            "Design,' Proceedings of i-SAIRAS 2020, paper #5068."
        ),
        notes=(
            "Concept paper from CMU / Astrobotic for the NASA LSITP-"
            "awarded polar micro-rover (Kumar et al. 2020 reports a "
            "13 kg full-up rover mass, 4 wheels, no RHU, lunar South "
            "Pole, single-daylight-period 8-Earth-day mission, "
            "max mechanical speed 0.07 m/s, rover length ~0.65 m, "
            "camera height 0.25 m).\n\n"
            "Mass-model schema convention: our DesignVector field "
            "``chassis_mass_kg`` is the structural-chassis-only mass "
            "(the bottom-up parametric mass model adds wheels, "
            "drive motors, solar, battery, avionics, harness, "
            "thermal, and a 25% growth margin on top). Kumar 2020's "
            "'13 kg' is the full-up total; we back-solve "
            "``chassis_mass_kg = 4.5`` (≈ 35 % of total) so the "
            "bottom-up sum matches the published 13 kg headline. "
            "This matches the convention in "
            "``data/mass_validation_set.csv`` and the rest of the "
            "rover_registry (Pragyan 38 %, Tenacious 40 %, CADRE "
            "40 %). Pre-fix (≤ 2026-05-26) the registry used "
            "``chassis_mass_kg=13.0`` and inflated the bottom-up "
            "total to ~25.7 kg; that was a registry bug, now "
            "corrected.\n\n"
            "Other imputations (class-match to Rashid-1 + polar "
            "power-budget back-solve): wheel radius/width, grouser "
            "height/count, solar area, battery, avionics, and peak "
            "wheel torque (v5-implicit anchor at 13 kg / 4 wheels / "
            "R = 0.10 m). The published 'max mechanical speed' is "
            "treated as a kinematic envelope (v_cruise is derived "
            "in our model), not a free design input, consistent "
            "with the v6 schema. See ``RoverRegistryEntry`` "
            "imputation_notes for the full list."
        ),
        # Published numbers from the paper:
        #
        # - range_km: Kumar 2020 targets ~1 km/Earth-day exploration
        #   over an 8-Earth-day mission, giving an 8 km headline
        #   traverse. The paper bands this 4-10 km depending on
        #   subsystem-margin assumptions; we cite the midpoint with
        #   that caveat documented above.
        # - total_mass_kg: 13 kg is the cited total. The
        #   DesignVector-driven bottom-up model produces a different
        #   number (see notes); we cite 13 kg here so the report
        #   shows the paper's headline side-by-side with our
        #   prediction.
        published_metrics={
            "range_km": 8.0,
            "total_mass_kg": 13.0,
        },
        # Pre-Phase-A class envelope. CMU/Astrobotic targeted the
        # 30-kg-class CLPS small-rover slot; 30 kg keeps the Pareto
        # front inside the lander payload limit the team actually
        # faced and is large enough to accommodate the bottom-up
        # mass model's predicted 25.7 kg total for the published
        # design vector (so the published point is feasible under
        # the constraint, which is the right framing for a
        # dominance comparison).
        mass_envelope_kg=30.0,
    ),
}
"""Catalogue of pre-Phase-A concept studies for Layer-6 comparison.

The current single entry is MoonRanger (Kumar et al. i-SAIRAS 2020).
VIPER is a deliberate stretch — its publicly available design
documents specify mass / wheels / power architecture but the full
11-D design vector is not cleanly consolidated in a single citable
paper, so we hold off until that's done rather than ship imputed
numbers without provenance.
"""


# ---------------------------------------------------------------------------
# Comparison machinery
# ---------------------------------------------------------------------------


def _evaluate_published_metrics(
    design: DesignVector,
    scenario: MissionScenario,
) -> dict[str, float]:
    """Run the analytical evaluator on a single design under a scenario."""
    metrics = evaluator_evaluate(design, scenario)
    return {
        "range_km": float(metrics.range_km),
        "energy_margin_raw_pct": float(metrics.energy_margin_raw_pct),
        "slope_capability_deg": float(metrics.slope_capability_deg),
        "total_mass_kg": float(metrics.total_mass_kg),
    }


def _best_per_objective(
    concept_metrics: Mapping[str, float],
    pareto_metrics: list[dict[str, float]],
    objectives: tuple[OptimizationObjective, ...],
) -> dict[str, dict[str, float]]:
    """Per-objective best-Pareto-point summary versus the concept's value.

    ``percent_delta`` is ``NaN`` when the concept's value rounds to
    zero (typically because the published design has an infeasible
    energy budget under our analytical physics). A percent change
    against zero is mathematically undefined; the absolute ``delta``
    remains the canonical headline in that case.
    """
    out: dict[str, dict[str, float]] = {}
    near_zero = 1e-3
    for obj in objectives:
        values = np.asarray([m[obj.target] for m in pareto_metrics], dtype=float)
        if obj.direction == "max":
            idx = int(np.argmax(values))
        else:
            idx = int(np.argmin(values))
        cand_val = float(values[idx])
        concept_val = float(concept_metrics[obj.target])
        delta = cand_val - concept_val
        if abs(concept_val) < near_zero:
            percent_delta = float("nan")
        else:
            percent_delta = delta / abs(concept_val) * 100.0
        out[obj.target] = {
            "pareto_index": float(idx),
            "value": cand_val,
            "concept_value": concept_val,
            "delta": delta,
            "percent_delta": percent_delta,
        }
    return out


def compare_concept_to_pareto(
    entry: ConceptStudyEntry,
    *,
    objectives: tuple[OptimizationObjective, ...] = DEFAULT_OBJECTIVES,
    backend: OptimizationBackend = "evaluator",
    bundles: dict[str, QuantileHeads] | None = None,
    population_size: int = 200,
    n_generations: int = 50,
    seed: int = 0,
    evaluator_eval_cap: int = 25_000,
) -> ConceptStudyComparison:
    """Run NSGA-II under the concept's scenario, compare to its published design.

    The optimiser is constrained only by the concept's stated
    ``mass_envelope_kg`` (default: no constraint). The leakage
    controls of Layer-5 rediscovery still apply because the
    scenario is always one of the class-generic ``*_micro`` YAMLs.

    Parameters
    ----------
    entry
        The :class:`ConceptStudyEntry` to compare. Drives the
        scenario, the published design vector, and the class
        envelope.
    objectives
        Pareto objectives. Default
        ``DEFAULT_OBJECTIVES = (range_km↑, total_mass_kg↓,
        slope_capability_deg↑)`` matches Layer-5 rediscovery so
        the two layers' tables are directly comparable.
    backend, bundles, evaluator_eval_cap
        Same semantics as :func:`rover_rediscovery.rediscover`.
        The analytical evaluator is *always* used for the head-to-
        head metrics on ``published_design`` so the comparison
        is well-defined regardless of the NSGA-II backend.
    population_size, n_generations, seed
        NSGA-II hyperparameters. The default 200 × 50 = 10 000
        evaluations is a high-budget run on the evaluator (under the
        25k safety cap) and trivially cheap on the surrogate.

    Returns
    -------
    ConceptStudyComparison
        Dominance counts, design-space proximity, per-objective
        deltas, and the full :class:`OptimizationResult`.
    """
    scenario = entry.load_scenario()
    soil = get_soil_parameters(scenario.soil_simulant)

    constraints: tuple[OptimizationConstraint, ...] = ()
    if entry.mass_envelope_kg is not None:
        constraints = (
            OptimizationConstraint(
                target="total_mass_kg",
                sense="max",
                value=float(entry.mass_envelope_kg),
            ),
        )

    runner = NSGA2Runner(
        scenario,
        soil,
        backend=backend,
        bundles=bundles,
        objectives=objectives,
        constraints=constraints,
        population_size=population_size,
        n_generations=n_generations,
        seed=seed,
        evaluator_eval_cap=evaluator_eval_cap,
    )
    opt_result = runner.run()

    if not opt_result.design_vectors:
        raise RuntimeError(
            f"NSGA-II returned an empty Pareto front for {entry.rover_name!r}; "
            f"every individual violated the mass envelope "
            f"{entry.mass_envelope_kg!r} kg under scenario {entry.scenario_name!r}."
        )

    # Always evaluate the published design with the analytical physics
    # so the head-to-head comparison is apples-to-apples regardless of
    # which backend NSGA-II used for its fitness function.
    published_metrics_predicted = _evaluate_published_metrics(
        entry.published_design, scenario
    )

    n_dom = sum(
        1
        for m in opt_result.metrics
        if _dominates(m, published_metrics_predicted, objectives)
    )
    n_dom_by_concept = sum(
        1
        for m in opt_result.metrics
        if _dominates(published_metrics_predicted, m, objectives)
    )

    distances = np.asarray(
        [_normalised_l2(d, entry.published_design) for d in opt_result.design_vectors],
        dtype=float,
    )
    nearest_idx = int(np.argmin(distances))
    nearest_dist = float(distances[nearest_idx])

    best_per_obj = _best_per_objective(
        published_metrics_predicted, opt_result.metrics, objectives
    )

    return ConceptStudyComparison(
        rover_name=entry.rover_name,
        scenario_name=entry.scenario_name,
        backend_used=backend,
        published_design=entry.published_design,
        published_metrics_predicted=published_metrics_predicted,
        published_metrics_cited=dict(entry.published_metrics),
        pareto_designs=list(opt_result.design_vectors),
        pareto_metrics=list(opt_result.metrics),
        n_pareto_points_dominating_concept=int(n_dom),
        n_pareto_points_dominated_by_concept=int(n_dom_by_concept),
        nearest_pareto_index=nearest_idx,
        nearest_pareto_design_space_distance=nearest_dist,
        best_per_objective=best_per_obj,
        optimization_result=opt_result,
    )


# ---------------------------------------------------------------------------
# Catalogue accessors
# ---------------------------------------------------------------------------


def list_concept_studies() -> list[str]:
    """Return the names of registered concept studies."""
    return sorted(CONCEPT_STUDIES.keys())


def get_concept_study(name: str) -> ConceptStudyEntry:
    """Look up one entry by name. Raises ``KeyError`` if absent."""
    if name not in CONCEPT_STUDIES:
        raise KeyError(
            f"unknown concept study {name!r}; known: {list_concept_studies()}"
        )
    return CONCEPT_STUDIES[name]


__all__ = [
    "CONCEPT_STUDIES",
    "ConceptStudyComparison",
    "ConceptStudyEntry",
    "compare_concept_to_pareto",
    "get_concept_study",
    "list_concept_studies",
]
