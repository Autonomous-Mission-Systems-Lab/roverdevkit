"""Validation harnesses for RoverDevKit.

- :mod:`.rover_registry`    — published-rover design vectors, scenarios,
  and truth numbers (real-rover validation).
- :mod:`.rover_comparison`  — run the evaluator on the registry and score
  vs truth (real-rover validation, Layer 4).
- :mod:`.rover_rediscovery` — the headline validation (Layer 5, rediscovery-validation).
- :mod:`.rediscovery_report` — LOO orchestration + report writer on
  top of :mod:`.rover_rediscovery` (Layer 5, paper-figure pipeline).
- :mod:`.concept_study_comparison` — head-to-head against published
  pre-Phase-A concept-study design points (Layer 6).
- :mod:`.cross_scenario`    — qualitative archetype-ranking checks across
  the four scenario families and one-at-a-time design-variable sensitivity
  (real-rover validation, Layer 7).
- :mod:`.robustness`        — Layer-6 ±20 % soil and ``MassModelParams``
  perturbation sweep over (scenario × archetype) cells; reports
  continuous-metric drift and ranking-preservation per scenario.

Layer-3 sub-model validation against published wheel-testbed data is
shipped as the parametric reference grid in
``data/validation/wong_layer3_reference.csv``, exercised by
``tests/test_terramechanics.py::test_layer3_published_reference_grid``
(Wong 2008 §4.2 worked-example fixture, Pragyan- and Yutu-2-class
rover-class operating points, and Iizuka & Kubota 2011 grouser-thrust
limit cases, each row with documented tolerance bands). The
consolidated layered error-budget that pulls Layers 1-7 into a single
chain lives at ``reports/error_budget.md``.
"""

from roverdevkit.validation.concept_study_comparison import (
    CONCEPT_STUDIES,
    ConceptStudyComparison,
    ConceptStudyEntry,
    compare_concept_to_pareto,
    get_concept_study,
    list_concept_studies,
)
from roverdevkit.validation.rediscovery_report import (
    DEFAULT_PER_ROVER_OVERRIDES,
    RediscoveryRunSummary,
    run_rediscovery_loo,
    summarize_results,
    write_loo_artifacts,
)
from roverdevkit.validation.robustness import (
    Perturbation,
    RobustnessEntry,
    RobustnessSummary,
    cross_scenario_robustness_sweep,
    default_perturbations,
    format_robustness_report,
    perturb_mass_params,
    perturb_soil,
)
from roverdevkit.validation.rover_comparison import (
    ComparisonSummary,
    RoverComparisonResult,
    acceptance_gate,
    compare_all,
    compare_one,
    format_report,
)
from roverdevkit.validation.rover_rediscovery import (
    RediscoveryResult,
    class_generic_scenario_for,
    rediscover,
    rediscover_all,
)
from roverdevkit.validation.rover_registry import (
    PublishedTruth,
    RoverRegistryEntry,
    flown_registry,
    load_truth_table,
    registry,
    registry_by_name,
    truth_by_rover,
)

__all__ = [
    "CONCEPT_STUDIES",
    "ComparisonSummary",
    "ConceptStudyComparison",
    "ConceptStudyEntry",
    "DEFAULT_PER_ROVER_OVERRIDES",
    "Perturbation",
    "PublishedTruth",
    "RediscoveryResult",
    "RediscoveryRunSummary",
    "RobustnessEntry",
    "RobustnessSummary",
    "RoverComparisonResult",
    "RoverRegistryEntry",
    "acceptance_gate",
    "class_generic_scenario_for",
    "compare_all",
    "compare_concept_to_pareto",
    "compare_one",
    "cross_scenario_robustness_sweep",
    "default_perturbations",
    "flown_registry",
    "format_report",
    "format_robustness_report",
    "get_concept_study",
    "list_concept_studies",
    "load_truth_table",
    "perturb_mass_params",
    "perturb_soil",
    "rediscover",
    "rediscover_all",
    "registry",
    "registry_by_name",
    "run_rediscovery_loo",
    "summarize_results",
    "truth_by_rover",
    "write_loo_artifacts",
]
