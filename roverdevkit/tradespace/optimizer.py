"""NSGA-II multi-objective optimization via pymoo.

The optimizer is deliberately small and webapp-agnostic: callers provide
the canonical scenario, soil parameters, optionally a set of loaded
quantile bundles, and optionally the evaluator's wheel-level SCM
correction artifact. The default backend is the corrected physics
evaluator: at ~20 ms per design it can finish a 1,500-evaluation
NSGA-II search in well under a minute, and using evaluator-truth as the
fitness function avoids any surrogate-approximation error on the
optimization frontier. The surrogate backend is retained as an opt-in
benchmarking option for callers that need sub-millisecond fitness
evaluations (e.g., large offline experiments).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.core.problem import Problem
from pymoo.indicators.hv import HV
from pymoo.optimize import minimize

from roverdevkit.mission.evaluator import evaluate as evaluator_evaluate
from roverdevkit.schema import DesignVector, MissionScenario
from roverdevkit.surrogate.features import (
    INPUT_COLUMNS,
    PRIMARY_REGRESSION_TARGETS,
    SCENARIO_CATEGORICAL_COLUMNS,
)
from roverdevkit.surrogate.uncertainty import QuantileHeads
from roverdevkit.terramechanics.bekker_wong import SoilParameters
from roverdevkit.terramechanics.correction_model import WheelLevelCorrection

ObjectiveDirection = Literal["min", "max"]
OptimizationBackend = Literal["surrogate", "evaluator"]

DESIGN_VARIABLES: tuple[str, ...] = (
    "wheel_radius_m",
    "wheel_width_m",
    "grouser_height_m",
    "grouser_count",
    "n_wheels",
    "chassis_mass_kg",
    "wheelbase_m",
    "solar_area_m2",
    "battery_capacity_wh",
    "avionics_power_w",
    "peak_wheel_torque_nm",
)
"""Design-vector field order used for the pymoo decision vector."""

DESIGN_BOUNDS: dict[str, tuple[float, float]] = {
    "wheel_radius_m": (0.05, 0.20),
    "wheel_width_m": (0.03, 0.20),
    "grouser_height_m": (0.0, 0.020),
    "grouser_count": (0.0, 24.0),
    "n_wheels": (4.0, 6.0),
    "chassis_mass_kg": (0.5, 50.0),
    "wheelbase_m": (0.3, 1.2),
    "solar_area_m2": (0.1, 1.5),
    "battery_capacity_wh": (5.0, 500.0),
    "avionics_power_w": (5.0, 40.0),
    "peak_wheel_torque_nm": (0.05, 20.0),
}
"""NSGA-II search bounds. Mirror :class:`DesignVector` schema bounds so
the optimiser can reach every constructable design. The three
ultra-micro floors (``chassis_mass_kg``, ``battery_capacity_wh``,
``peak_wheel_torque_nm``) were lowered 2026-05-27 to admit CADRE and
Tenacious. The v4 LHS surrogate was trained on the narrower
``(3.0, 20.0, 0.3)`` floors; running NSGA-II with ``backend='surrogate'``
on designs below those points extrapolates outside training support
until the v5 regeneration."""


@dataclass(frozen=True)
class OptimizationObjective:
    """One Pareto objective over a primary mission metric."""

    target: str
    direction: ObjectiveDirection

    def __post_init__(self) -> None:
        if self.target not in PRIMARY_REGRESSION_TARGETS:
            raise ValueError(
                f"target {self.target!r} is not optimizable; "
                f"allowed: {PRIMARY_REGRESSION_TARGETS}."
            )


@dataclass(frozen=True)
class OptimizationConstraint:
    """Scalar threshold constraint over a primary mission metric."""

    target: str
    sense: Literal["min", "max"]
    value: float

    def __post_init__(self) -> None:
        if self.target not in PRIMARY_REGRESSION_TARGETS:
            raise ValueError(
                f"constraint target {self.target!r} is not supported; "
                f"allowed: {PRIMARY_REGRESSION_TARGETS}."
            )


@dataclass(frozen=True)
class OptimizationCheckpoint:
    """Per-generation progress snapshot emitted by :class:`NSGA2Runner`."""

    gen: int
    hypervolume: float
    pareto_size: int
    best_per_objective: dict[str, float]


@dataclass(frozen=True)
class OptimizationResult:
    """Final Pareto front returned by :class:`NSGA2Runner`."""

    design_vectors: list[DesignVector]
    metrics: list[dict[str, float]]
    objectives: tuple[OptimizationObjective, ...]
    backend_used: OptimizationBackend
    checkpoints: list[OptimizationCheckpoint] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        """Return one row per Pareto point with design fields + metrics."""
        rows: list[dict[str, float | int | str]] = []
        for design, metric in zip(self.design_vectors, self.metrics, strict=True):
            row: dict[str, float | int | str] = dict(design.model_dump())
            row.update({k: float(v) for k, v in metric.items()})
            row["backend_used"] = self.backend_used
            rows.append(row)
        return pd.DataFrame(rows)


DEFAULT_OBJECTIVES: tuple[OptimizationObjective, ...] = (
    OptimizationObjective("range_km", "max"),
    OptimizationObjective("total_mass_kg", "min"),
    OptimizationObjective("slope_capability_deg", "max"),
)


class NSGA2Runner:
    """Run NSGA-II over the rover design space."""

    def __init__(
        self,
        scenario: MissionScenario,
        soil: SoilParameters,
        *,
        bundles: dict[str, QuantileHeads] | None = None,
        correction: WheelLevelCorrection | None = None,
        backend: OptimizationBackend = "evaluator",
        objectives: tuple[OptimizationObjective, ...] = DEFAULT_OBJECTIVES,
        constraints: tuple[OptimizationConstraint, ...] = (),
        population_size: int = 100,
        n_generations: int = 200,
        seed: int = 0,
        evaluator_eval_cap: int = 1000,
        panel_tilt_deg: float = 0.0,
        panel_azimuth_deg: float = 180.0,
    ) -> None:
        """Construct a runner.

        Parameters
        ----------
        panel_tilt_deg, panel_azimuth_deg
            Solar-array orientation forwarded to every evaluator call
            (and so to :func:`roverdevkit.mission.traverse_sim.run_traverse`).
            Defaults match the simulator's historical horizontal /
            south-facing panel. The rediscovery harness sets these to
            a scenario-driven ``tilt = min(80, |latitude|)`` /
            sun-tracking azimuth at high latitudes so the optimiser's
            Pareto front is evaluated under the same panel-pointing
            assumption as the real polar rovers it is being compared
            against. Note: only the evaluator backend honours these
            overrides; the surrogate backend is trained on
            horizontal-panel evaluator outputs and ignores tilt /
            azimuth (a v9 LHS regen would be required to restore
            symmetry at high latitudes).
        """
        if backend == "surrogate" and bundles is None:
            raise ValueError("surrogate backend requires quantile bundles.")
        if backend == "evaluator" and population_size * n_generations > evaluator_eval_cap:
            raise ValueError(
                f"evaluator backend is capped at {evaluator_eval_cap} evaluations "
                f"(requested {population_size * n_generations})."
            )
        self.scenario = scenario
        self.soil = soil
        self.bundles = bundles
        self.correction = correction
        self.backend = backend
        self.objectives = objectives
        self.constraints = constraints
        self.population_size = population_size
        self.n_generations = n_generations
        self.seed = seed
        self.panel_tilt_deg = panel_tilt_deg
        self.panel_azimuth_deg = panel_azimuth_deg

    def run(
        self,
        *,
        on_checkpoint: Callable[[OptimizationCheckpoint], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> OptimizationResult:
        """Run NSGA-II and return the final non-dominated front."""
        problem = _RoverProblem(self)
        checkpoints: list[OptimizationCheckpoint] = []

        def record(checkpoint: OptimizationCheckpoint) -> None:
            checkpoints.append(checkpoint)
            if on_checkpoint is not None:
                on_checkpoint(checkpoint)

        algorithm = NSGA2(pop_size=self.population_size)
        result = minimize(
            problem,
            algorithm,
            ("n_gen", self.n_generations),
            seed=self.seed,
            callback=_CheckpointCallback(
                objectives=self.objectives,
                record=record,
                should_cancel=should_cancel,
            ),
            verbose=False,
            save_history=False,
        )

        if result.X is None:
            return OptimizationResult(
                design_vectors=[],
                metrics=[],
                objectives=self.objectives,
                backend_used=self.backend,
                checkpoints=checkpoints,
            )
        X = np.atleast_2d(result.X)
        designs = [_vector_to_design(row) for row in X]
        metrics = self._evaluate_designs(designs)
        return OptimizationResult(
            design_vectors=designs,
            metrics=metrics,
            objectives=self.objectives,
            backend_used=self.backend,
            checkpoints=checkpoints,
        )

    def _evaluate_designs(self, designs: list[DesignVector]) -> list[dict[str, float]]:
        if self.backend == "surrogate":
            if self.bundles is None:  # pragma: no cover - constructor guards this
                raise AssertionError("missing bundles")
            return _surrogate_metrics(designs, self.scenario, self.soil, self.bundles)
        return _evaluator_metrics(
            designs,
            self.scenario,
            correction=self.correction,
            panel_tilt_deg=self.panel_tilt_deg,
            panel_azimuth_deg=self.panel_azimuth_deg,
        )


class _RoverProblem(Problem):
    """pymoo vectorized problem wrapper."""

    def __init__(self, runner: NSGA2Runner) -> None:
        xl = np.array([DESIGN_BOUNDS[name][0] for name in DESIGN_VARIABLES], dtype=float)
        xu = np.array([DESIGN_BOUNDS[name][1] for name in DESIGN_VARIABLES], dtype=float)
        super().__init__(
            n_var=len(DESIGN_VARIABLES),
            n_obj=len(runner.objectives),
            n_ieq_constr=len(runner.constraints),
            xl=xl,
            xu=xu,
        )
        self.runner = runner

    def _evaluate(self, X: np.ndarray, out: dict[str, np.ndarray], *args: object, **kwargs: object) -> None:
        designs = [_vector_to_design(row) for row in np.atleast_2d(X)]
        metrics = self.runner._evaluate_designs(designs)
        out["F"] = np.asarray(
            [
                [
                    _objective_value(metric[obj.target], obj.direction)
                    for obj in self.runner.objectives
                ]
                for metric in metrics
            ],
            dtype=float,
        )
        if self.runner.constraints:
            out["G"] = np.asarray(
                [
                    [
                        _constraint_violation(metric[constraint.target], constraint)
                        for constraint in self.runner.constraints
                    ]
                    for metric in metrics
                ],
                dtype=float,
            )


class _CheckpointCallback(Callback):
    def __init__(
        self,
        *,
        objectives: tuple[OptimizationObjective, ...],
        record: Callable[[OptimizationCheckpoint], None],
        should_cancel: Callable[[], bool] | None,
    ) -> None:
        super().__init__()
        self.objectives = objectives
        self.record = record
        self.should_cancel = should_cancel

    def notify(self, algorithm: object) -> None:
        pop = algorithm.pop  # type: ignore[attr-defined]
        F = np.asarray(pop.get("F"), dtype=float)
        if F.size == 0:
            return
        feasible = _feasible_mask(pop)
        front = F[feasible] if np.any(feasible) else F
        checkpoint = OptimizationCheckpoint(
            gen=int(algorithm.n_gen),  # type: ignore[attr-defined]
            hypervolume=_hypervolume(front),
            pareto_size=int(front.shape[0]),
            best_per_objective=_best_per_objective(front, self.objectives),
        )
        self.record(checkpoint)
        if self.should_cancel is not None and self.should_cancel():
            algorithm.termination.force_termination = True  # type: ignore[attr-defined]


def _vector_to_design(x: np.ndarray) -> DesignVector:
    values = {name: float(value) for name, value in zip(DESIGN_VARIABLES, x, strict=True)}
    values["grouser_count"] = int(np.clip(round(values["grouser_count"]), 0, 24))
    values["n_wheels"] = 4 if values["n_wheels"] < 5.0 else 6
    return DesignVector(**values)


def _feature_frame(
    designs: list[DesignVector],
    scenario: MissionScenario,
    soil: SoilParameters,
) -> pd.DataFrame:
    rows = []
    for design in designs:
        rows.append(
            {
                "design_wheel_radius_m": design.wheel_radius_m,
                "design_wheel_width_m": design.wheel_width_m,
                "design_grouser_height_m": design.grouser_height_m,
                "design_grouser_count": int(design.grouser_count),
                "design_n_wheels": int(design.n_wheels),
                "design_chassis_mass_kg": design.chassis_mass_kg,
                "design_wheelbase_m": design.wheelbase_m,
                "design_solar_area_m2": design.solar_area_m2,
                "design_battery_capacity_wh": design.battery_capacity_wh,
                "design_avionics_power_w": design.avionics_power_w,
                "design_peak_wheel_torque_nm": design.peak_wheel_torque_nm,
                "scenario_latitude_deg": scenario.latitude_deg,
                "scenario_mission_duration_earth_days": scenario.mission_duration_earth_days,
                "scenario_max_slope_deg": scenario.max_slope_deg,
                "scenario_operational_duty_cycle": scenario.operational_duty_cycle,
                "scenario_soil_n": soil.n,
                "scenario_soil_k_c": soil.k_c,
                "scenario_soil_k_phi": soil.k_phi,
                "scenario_soil_cohesion_kpa": soil.cohesion_kpa,
                "scenario_soil_friction_angle_deg": soil.friction_angle_deg,
                "scenario_soil_shear_modulus_k_m": soil.shear_modulus_k_m,
                "scenario_payload_mass_kg": scenario.payload_mass_kg,
                "scenario_payload_power_w": scenario.payload_power_w,
                "scenario_family": scenario.name,
                "scenario_terrain_class": scenario.terrain_class,
                "scenario_soil_simulant": scenario.soil_simulant,
                "scenario_sun_geometry": scenario.sun_geometry,
            }
        )
    df = pd.DataFrame(rows, columns=INPUT_COLUMNS)
    for col in SCENARIO_CATEGORICAL_COLUMNS:
        df[col] = df[col].astype("category")
    return df


def _surrogate_metrics(
    designs: list[DesignVector],
    scenario: MissionScenario,
    soil: SoilParameters,
    bundles: dict[str, QuantileHeads],
) -> list[dict[str, float]]:
    missing = [target for target in PRIMARY_REGRESSION_TARGETS if target not in bundles]
    if missing:
        raise KeyError(f"quantile bundles missing targets: {missing}")
    X = _feature_frame(designs, scenario, soil)
    columns: dict[str, np.ndarray] = {}
    for target in PRIMARY_REGRESSION_TARGETS:
        preds = bundles[target].predict(X, repair_crossings=True)
        columns[target] = np.asarray(preds.get("q50"), dtype=float)
    return [
        {target: float(columns[target][i]) for target in PRIMARY_REGRESSION_TARGETS}
        for i in range(len(designs))
    ]


def _evaluator_metrics(
    designs: list[DesignVector],
    scenario: MissionScenario,
    *,
    correction: WheelLevelCorrection | None,
    panel_tilt_deg: float = 0.0,
    panel_azimuth_deg: float = 180.0,
) -> list[dict[str, float]]:
    """Evaluate a batch of designs under the corrected evaluator.

    A single evaluator failure (e.g. the Bekker-Wong slip solver
    cannot find an entry angle for a fully-buried wheel) must not
    crash the entire NSGA-II run. We catch the exception here and
    emit a sentinel "deeply infeasible" metrics dict that the
    objective/constraint pipeline will translate into a large
    constraint-violation magnitude, letting the GA continue.
    """
    out: list[dict[str, float]] = []
    use_corr = correction is not None
    for design in designs:
        try:
            metrics = evaluator_evaluate(
                design,
                scenario,
                use_scm_correction=use_corr,
                correction=correction,
                panel_tilt_deg=panel_tilt_deg,
                panel_azimuth_deg=panel_azimuth_deg,
            )
            out.append(
                {
                    "range_km": float(metrics.range_km),
                    "energy_margin_raw_pct": float(metrics.energy_margin_raw_pct),
                    "slope_capability_deg": float(metrics.slope_capability_deg),
                    "total_mass_kg": float(metrics.total_mass_kg),
                }
            )
        except Exception:  # noqa: BLE001 -- broad-except is intentional
            # Sentinel: zero range/slope, large mass so any constraint
            # using total_mass_kg as a ceiling treats this as
            # infeasible, and the GA's tournament selection prefers
            # any successfully-evaluated individual.
            out.append(
                {
                    "range_km": 0.0,
                    "energy_margin_raw_pct": -100.0,
                    "slope_capability_deg": 0.0,
                    "total_mass_kg": 1e6,
                }
            )
    return out


def _objective_value(value: float, direction: ObjectiveDirection) -> float:
    return float(value if direction == "min" else -value)


def _constraint_violation(value: float, constraint: OptimizationConstraint) -> float:
    if constraint.sense == "min":
        return float(constraint.value - value)
    return float(value - constraint.value)


def _feasible_mask(pop: object) -> np.ndarray:
    try:
        cv = np.asarray(pop.get("CV"), dtype=float)
        if cv.ndim == 2:
            cv = cv[:, 0]
        return cv <= 0.0
    except Exception:
        return np.ones(len(pop), dtype=bool)  # type: ignore[arg-type]


def _hypervolume(F: np.ndarray) -> float:
    if F.ndim != 2 or F.shape[0] == 0:
        return 0.0
    finite = F[np.all(np.isfinite(F), axis=1)]
    if finite.size == 0:
        return 0.0
    ref = np.nanmax(finite, axis=0) + 1.0
    try:
        return float(HV(ref_point=ref).do(finite))
    except Exception:
        return 0.0


def _best_per_objective(
    F: np.ndarray, objectives: tuple[OptimizationObjective, ...]
) -> dict[str, float]:
    out: dict[str, float] = {}
    for i, obj in enumerate(objectives):
        values = F[:, i]
        best_minimized = float(np.nanmin(values))
        out[obj.target] = best_minimized if obj.direction == "min" else -best_minimized
    return out


def run_nsga2(
    scenario: MissionScenario,
    soil: SoilParameters,
    *,
    bundles: dict[str, QuantileHeads] | None = None,
    correction: WheelLevelCorrection | None = None,
    backend: OptimizationBackend = "surrogate",
    objectives: tuple[OptimizationObjective, ...] = DEFAULT_OBJECTIVES,
    constraints: tuple[OptimizationConstraint, ...] = (),
    population_size: int = 100,
    n_generations: int = 200,
    seed: int = 0,
    panel_tilt_deg: float = 0.0,
    panel_azimuth_deg: float = 180.0,
) -> pd.DataFrame:
    """Run NSGA-II and return the Pareto-front design-and-metric dataframe."""
    runner = NSGA2Runner(
        scenario,
        soil,
        bundles=bundles,
        correction=correction,
        backend=backend,
        objectives=objectives,
        constraints=constraints,
        population_size=population_size,
        n_generations=n_generations,
        seed=seed,
        panel_tilt_deg=panel_tilt_deg,
        panel_azimuth_deg=panel_azimuth_deg,
    )
    return runner.run().to_frame()


__all__ = [
    "DEFAULT_OBJECTIVES",
    "DESIGN_BOUNDS",
    "DESIGN_VARIABLES",
    "NSGA2Runner",
    "OptimizationBackend",
    "OptimizationCheckpoint",
    "OptimizationConstraint",
    "OptimizationObjective",
    "OptimizationResult",
    "run_nsga2",
]
