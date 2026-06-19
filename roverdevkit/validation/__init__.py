"""Validation harnesses for RoverDevKit.

- :mod:`.rover_registry`    — published-rover design vectors, scenarios,
  and truth numbers (real-rover validation).
- :mod:`.rover_comparison`  — run the evaluator on the registry and score
  vs truth (real-rover validation, Layer 4).
- :mod:`.rover_rediscovery` — the headline validation (Layer 5, rediscovery-validation).
- :mod:`.rediscovery_report` — rediscovery orchestration + report writer on
  top of :mod:`.rover_rediscovery` (Layer 5, paper-figure pipeline).

Layer-3 sub-model validation against published wheel-testbed data has two
strands:

- :mod:`.terramechanics_experiment` — experiment-vs-model harness comparing
  the BW kernel to *measured* single-wheel
  drawbar-pull / sinkage / torque digitised from published experiments
  (worksheet ``data/validation/single_wheel_experiments.csv``; figure
  ``reports/figures/fig_terramechanics_experiment.png``; tested by
  ``tests/test_terramechanics_experiment.py``).
- the parametric reference grid in
  ``data/validation/wong_layer3_reference.csv``, exercised by
  ``tests/test_terramechanics.py::test_layer3_published_reference_grid``
  (Wong 2008 §4.2 worked-example fixture, Pragyan- and Yutu-2-class
  rover-class operating points, and Iizuka & Kubota 2011 grouser-thrust
  limit cases, each row with documented tolerance bands).

The consolidated layered error-budget that pulls Layers 1-5 into a single
chain lives at ``reports/error_budget.md``.
"""

from roverdevkit.validation.rediscovery_report import (
    DEFAULT_PER_ROVER_OVERRIDES,
    RediscoveryRunSummary,
    run_rediscovery_loo,
    summarize_results,
    write_loo_artifacts,
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
from roverdevkit.validation.terramechanics_experiment import (
    ExperimentPoint,
    compare_to_experiment,
    load_experiment_points,
    summarise,
)

__all__ = [
    "ComparisonSummary",
    "DEFAULT_PER_ROVER_OVERRIDES",
    "ExperimentPoint",
    "PublishedTruth",
    "RediscoveryResult",
    "RediscoveryRunSummary",
    "RoverComparisonResult",
    "RoverRegistryEntry",
    "acceptance_gate",
    "class_generic_scenario_for",
    "compare_all",
    "compare_one",
    "compare_to_experiment",
    "flown_registry",
    "format_report",
    "load_experiment_points",
    "load_truth_table",
    "rediscover",
    "rediscover_all",
    "registry",
    "registry_by_name",
    "run_rediscovery_loo",
    "summarise",
    "summarize_results",
    "truth_by_rover",
    "write_loo_artifacts",
]
