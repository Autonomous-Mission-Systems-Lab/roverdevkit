"""Stratified Latin-Hypercube sampler for the Phase-2 analytical dataset.

Produces ``(DesignVector, MissionScenario, SoilParameters)`` triples for
the surrogate training set (``project_plan.md`` §6). The sampler is
deterministic given its seed; re-running with the same ``seed`` reproduces
the exact same set of ``(design, scenario)`` pairs and their train/val/
test split assignment.

Sampling strategy
-----------------
Two orthogonal stratifications are applied:

1. **Scenario family** — each of the four canonical scenarios
   (``equatorial_mare_traverse``, ``polar_prospecting``,
   ``highland_slope_capability``, ``crater_rim_survey``) gets its own
   LHS sweep of size ``n_per_scenario``. Scenario-level parameters
   (latitude, mission duration, max slope, Bekker soil params) are
   jittered *within* the family so the surrogate learns a continuous
   cross-scenario mapping instead of a four-category one, while still
   respecting per-family soil/sun-geometry realism.
2. **Wheel count** — within each scenario-family sweep, samples are
   split 50/50 between 4-wheel and 6-wheel designs. ``n_wheels`` is
   categorical and cannot be swept by LHS, so stratifying here keeps
   both strata equally represented without doubling the sample count.

Continuous design variables (10) and scenario-level perturbation
variables (3 mission + 1 ops duty + 6 soil = 10) are stacked into a
single (10 + 10)-D LHS across each stratum, then unscaled to their
physical ranges. ``grouser_count`` is drawn from a continuous LHS
column and rounded to an integer in ``[0, 24]``.

Schema-version note (v7_1, W12 step B follow-on): the
``operational_duty_cycle`` column is now drawn from the LHS over
[0, 0.6] independently of the scenario family, instead of being
pinned to the family default. The per-family default is retained on
:class:`ScenarioFamily` for canonical YAML / UI initial slider use
but the *training distribution* is family-agnostic so the calibrated
quantile heads are valid across the full frontend slider range.

Split assignment
----------------
Each sample is tagged with ``split ∈ {train, val, test}`` *at sample
generation time* using a per-index RNG seeded from the main seed. The
split is therefore stable across runs and does **not** depend on
whether a particular evaluation succeeded or failed: the hold-out
distribution is fixed before any physics runs.

Output
------
:func:`generate_samples` returns a list of :class:`LHSSample` with the
fully materialised :class:`DesignVector` and :class:`MissionScenario`
already validated (pydantic) plus a :class:`SoilParameters` to pass to
:func:`evaluator.evaluate` as ``soil_override``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import qmc

from roverdevkit.drivetrain.motor import sizing_peak_torque_anchor_nm
from roverdevkit.schema import DesignVector, MissionScenario, TerrainClass
from roverdevkit.terramechanics.bekker_wong import SoilParameters

# ---------------------------------------------------------------------------
# Design-variable bounds (mirror DesignVector field constraints)
# ---------------------------------------------------------------------------

SplitName = Literal["train", "val", "test"]


# Continuous design variables swept by LHS (order matters; see _unscale_design).
#
# Bounds widened in SCHEMA_VERSION v3 (2026-04-25) for `wheel_width_m`,
# `grouser_height_m`, and `chassis_mass_kg` so the flown / design-target
# lunar micro-rovers in `roverdevkit.validation.rover_registry` sit
# inside the surrogate's training support rather than at corner points
# of the cube. See project_log.md for the rationale and the registry
# entries (Yutu-2 mass ~35 kg ex-payload, Rashid-1 grouser 15 mm,
# Lunokhod-class wheel widths 20 cm) that motivated the widening.
# SCHEMA_VERSION v6 (2026-04-28, W12 step B): ``nominal_speed_mps`` is
# no longer a free design variable (cruise speed is now derived inside
# the evaluator from drivetrain torque + slip-balance + energy
# balance + kinematic envelope) and ``drive_duty_cycle`` is renamed
# ``designed_duty_cycle`` (the "sizing" half of the duty semantics;
# see ``MissionScenario.operational_duty_cycle`` for the ground-ops
# half). ``peak_wheel_torque_nm`` enters as a true drivetrain
# capability input. The LHS column ``peak_wheel_torque_nm`` is
# sampled in *log-space* (anchored at the v5-implicit hub torque,
# log-uniform [0.5, 3.0]; clipped to schema bounds) by
# :func:`_build_design_from_lhs_row`, not via this uniform-bound
# entry — the schema bounds here are the floor / ceiling clips, not
# the prior shape.
#
# SCHEMA_VERSION v7 (2026-04-28, W12 step B follow-up): drops
# ``designed_duty_cycle`` from the LHS bounds tuple after that field
# turned out to do no engineering work in the v6 mass model. The
# only role of δ_des in v6 was to upper-bound δ_eff = min(δ_des,
# δ_ops); a user can equivalently express that by lowering δ_ops.
# Drive duty cycle now lives entirely on the scenario.
_CONTINUOUS_DESIGN_BOUNDS: tuple[tuple[str, float, float], ...] = (
    ("wheel_radius_m", 0.05, 0.20),
    ("wheel_width_m", 0.03, 0.20),
    ("grouser_height_m", 0.0, 0.020),
    ("chassis_mass_kg", 3.0, 50.0),
    ("wheelbase_m", 0.3, 1.2),
    ("solar_area_m2", 0.1, 1.5),
    ("battery_capacity_wh", 20.0, 500.0),
    ("avionics_power_w", 5.0, 40.0),
    ("peak_wheel_torque_nm", 0.3, 20.0),
)

# grouser_count is an integer LHS column
_GROUSER_COUNT_BOUNDS: tuple[int, int] = (0, 24)

# Scenario-level perturbation columns (per-family base values live in FAMILIES).
#
# SCHEMA_VERSION v7_1 (W12 step B follow-on, 2026-04-28): added
# ``operational_duty_cycle`` so the surrogate sees δ_ops as a true LHS
# feature instead of a per-family constant. The pre-v7_1 dataset
# pinned δ_ops to the family's published default (mare 0.30, polar
# 0.05, highland 0.15, crater 0.20), which made the
# ``operational_duty_cycle`` slider in the webapp fall through to
# evaluator-only mode for off-default values (no PIs). Sampling δ_ops
# uniformly over its schema bounds [0, 0.6] *independently of family*
# closes that gap: the user can pick any δ_ops on any scenario and
# the calibrated quantile heads still apply. The per-family default
# is retained on :class:`ScenarioFamily` because the canonical
# scenario YAMLs / UI initial slider position still reference it.
_SCENARIO_PERTURB_COLS: tuple[str, ...] = (
    "latitude_deg",
    "mission_duration_earth_days",
    "max_slope_deg",
    "operational_duty_cycle",
    "soil_n",
    "soil_k_c",
    "soil_k_phi",
    "soil_cohesion_kpa",
    "soil_friction_angle_deg",
    "soil_shear_modulus_k_m",
)

# Per-row δ_ops bounds for the LHS draw. Matches
# :attr:`MissionScenario.operational_duty_cycle` schema bounds; chosen
# so the entire frontend slider range is in-distribution.
_OPERATIONAL_DUTY_CYCLE_BOUNDS: tuple[float, float] = (0.0, 0.6)

# Unified soil parameter bounds, covering the envelope of the seven
# simulants in data/soil_simulants.csv. The LHS draws jittered Bekker
# parameters in these ranges so the surrogate learns a continuous
# soil-property -> metric mapping (project_plan.md §6). The
# scenario-family labels retain physical realism because the terrain
# class / soil-simulant *name* is still attached to each sample, but
# the actual Bekker numbers the evaluator sees are the LHS draw.
_SOIL_BOUNDS: dict[str, tuple[float, float]] = {
    "soil_n": (0.8, 1.2),
    "soil_k_c": (0.5, 2.0),
    "soil_k_phi": (400.0, 1200.0),
    "soil_cohesion_kpa": (0.1, 1.0),
    "soil_friction_angle_deg": (30.0, 50.0),
    "soil_shear_modulus_k_m": (0.010, 0.025),
}


# ---------------------------------------------------------------------------
# Scenario families
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScenarioFamily:
    """One of the four canonical tradespace scenarios.

    Family-fixed attributes (terrain class, nominal soil simulant name,
    traverse-distance budget, sun geometry) are carried alongside the
    LHS-jittered ranges for latitude, mission duration, and max slope.
    """

    name: str
    terrain_class: TerrainClass
    soil_simulant: str
    traverse_distance_m: float
    sun_geometry: Literal["continuous", "diurnal", "polar_intermittent"]
    latitude_range_deg: tuple[float, float]
    mission_duration_range_days: tuple[float, float]
    max_slope_range_deg: tuple[float, float]
    operational_duty_cycle: float
    """Per-family default ground-ops duty cycle. Schema v6 (W12 step B):
    each generated :class:`MissionScenario` carries the calibrated δ_ops
    for its family so the LHS dataset sees the same operational anchor
    the canonical YAMLs expose at runtime (mare 0.30, polar 0.05,
    highland 0.15, crater 0.20)."""


FAMILIES: dict[str, ScenarioFamily] = {
    "equatorial_mare_traverse": ScenarioFamily(
        name="equatorial_mare_traverse",
        terrain_class="mare_nominal",
        soil_simulant="Apollo_regolith_nominal",
        traverse_distance_m=80000.0,
        sun_geometry="diurnal",
        latitude_range_deg=(10.0, 25.0),
        mission_duration_range_days=(10.0, 18.0),
        max_slope_range_deg=(3.0, 18.0),
        operational_duty_cycle=0.30,
    ),
    "polar_prospecting": ScenarioFamily(
        name="polar_prospecting",
        terrain_class="polar_regolith",
        soil_simulant="Apollo_regolith_nominal",
        traverse_distance_m=30000.0,
        sun_geometry="polar_intermittent",
        latitude_range_deg=(-88.0, -80.0),
        mission_duration_range_days=(25.0, 35.0),
        max_slope_range_deg=(10.0, 28.0),
        operational_duty_cycle=0.05,
    ),
    "highland_slope_capability": ScenarioFamily(
        name="highland_slope_capability",
        terrain_class="highland_dense",
        soil_simulant="Apollo_regolith_loose",
        traverse_distance_m=20000.0,
        sun_geometry="diurnal",
        latitude_range_deg=(5.0, 20.0),
        mission_duration_range_days=(5.0, 10.0),
        max_slope_range_deg=(18.0, 30.0),
        operational_duty_cycle=0.15,
    ),
    "crater_rim_survey": ScenarioFamily(
        name="crater_rim_survey",
        terrain_class="mare_nominal",
        soil_simulant="Apollo_regolith_nominal",
        traverse_distance_m=25000.0,
        sun_geometry="diurnal",
        latitude_range_deg=(-15.0, 15.0),
        mission_duration_range_days=(3.0, 7.0),
        max_slope_range_deg=(10.0, 25.0),
        operational_duty_cycle=0.20,
    ),
}


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LHSSample:
    """One LHS draw ready to be passed to ``evaluator.evaluate_verbose``.

    ``soil`` is the *jittered* Bekker soil to use in place of the
    catalogue lookup; pass it to ``evaluate_verbose(..., soil_override=
    sample.soil)``. ``split``, ``stratum_id``, and ``sample_index`` are
    metadata the dataset writer copies straight into the parquet.
    """

    sample_index: int
    split: SplitName
    stratum_id: int  # 0 = 4-wheel, 1 = 6-wheel
    scenario_family: str
    design: DesignVector
    scenario: MissionScenario
    soil: SoilParameters


# ---------------------------------------------------------------------------
# LHS-to-physical unscaling helpers
# ---------------------------------------------------------------------------


def _unit_lhs(n: int, d: int, seed: int) -> np.ndarray:
    """Draw an ``(n, d)`` unit-cube LHS array with the given seed."""
    sampler = qmc.LatinHypercube(d=d, seed=seed, scramble=True)
    return sampler.random(n=n)


def _unscale(u: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return lo + u * (hi - lo)


def _assign_splits(n: int, seed: int, val_frac: float, test_frac: float) -> np.ndarray:
    """Deterministically assign train/val/test to ``n`` samples.

    Uses a dedicated RNG (seed ^ 0xBEEF) so the split does not depend on
    the LHS ordering. Returns an ndarray of str labels.
    """
    if val_frac < 0.0 or test_frac < 0.0 or val_frac + test_frac >= 1.0:
        raise ValueError(
            f"val_frac ({val_frac}) and test_frac ({test_frac}) must be >= 0 and sum < 1."
        )
    rng = np.random.default_rng(seed ^ 0xBEEF)
    train_frac = 1.0 - val_frac - test_frac
    return rng.choice(
        np.array(["train", "val", "test"]),
        size=n,
        p=[train_frac, val_frac, test_frac],
    )


# Schema bounds for ``peak_wheel_torque_nm``; the LHS column is mapped
# log-uniform around a per-row anchor (see ``_build_design_from_lhs_row``)
# rather than uniform on these bounds, so we keep them as a separate
# clip rather than driving the unscale.
_PEAK_TORQUE_NM_FLOOR: float = 0.3
_PEAK_TORQUE_NM_CEILING: float = 20.0
# Log-uniform range applied to the v5-implicit anchor (decision.md
# §"LHS prior on peak_wheel_torque_nm"): a row's anchor is multiplied
# by a draw from LogUniform(0.5, 3.0) before clipping.
_PEAK_TORQUE_LOGU_LO: float = 0.5
_PEAK_TORQUE_LOGU_HI: float = 3.0


def _peak_wheel_torque_anchor_for_row(
    chassis_mass_kg: float,
    wheel_radius_m: float,
    n_wheels: int,
) -> float:
    """Coarse v5-implicit per-wheel torque anchor for the LHS prior.

    Uses the same expression the v5 mass model used (safety factor ×
    friction × per-wheel weight × radius), but with a coarse total-mass
    estimate (``2.5 × chassis_mass``) since motor mass is not yet
    sized. This is a *prior anchor only* — it is multiplied by a
    LogUniform(0.5, 3.0) factor and clipped to schema bounds before
    being written to the design vector. Designed so most LHS rows land
    near a physically realisable torque sizing for the rest of their
    design vector, avoiding an LHS that spends most of its samples in
    grossly under- or over-sized motor regimes.
    """
    coarse_total_mass = 2.5 * chassis_mass_kg
    return sizing_peak_torque_anchor_nm(
        total_mass_kg=coarse_total_mass,
        wheel_radius_m=wheel_radius_m,
        n_wheels=n_wheels,
    )


def _build_design_from_lhs_row(
    u_continuous: np.ndarray,
    u_grouser: float,
    n_wheels: int,
) -> DesignVector:
    """Convert one unit-cube LHS row to a validated :class:`DesignVector`.

    SCHEMA_VERSION v6: ``peak_wheel_torque_nm`` is sampled
    log-uniform around the per-row v5-implicit hub torque anchor (see
    :func:`_peak_wheel_torque_anchor_for_row`) rather than uniform on
    its schema bounds. All other continuous design variables are
    uniform-LHS as before.
    """
    kwargs: dict[str, float | int] = {}
    peak_torque_u: float | None = None
    for (name, lo, hi), u in zip(_CONTINUOUS_DESIGN_BOUNDS, u_continuous, strict=True):
        if name == "peak_wheel_torque_nm":
            peak_torque_u = float(u)
            continue
        kwargs[name] = float(_unscale(np.array([u]), lo, hi)[0])
    g_lo, g_hi = _GROUSER_COUNT_BOUNDS
    kwargs["grouser_count"] = int(round(g_lo + u_grouser * (g_hi - g_lo)))
    kwargs["n_wheels"] = n_wheels

    assert peak_torque_u is not None, (
        "peak_wheel_torque_nm must be present in _CONTINUOUS_DESIGN_BOUNDS; "
        "see SCHEMA_VERSION v6 in the bounds tuple comment."
    )
    anchor_nm = _peak_wheel_torque_anchor_for_row(
        chassis_mass_kg=float(kwargs["chassis_mass_kg"]),
        wheel_radius_m=float(kwargs["wheel_radius_m"]),
        n_wheels=n_wheels,
    )
    log_factor = _PEAK_TORQUE_LOGU_LO * (
        _PEAK_TORQUE_LOGU_HI / _PEAK_TORQUE_LOGU_LO
    ) ** peak_torque_u
    kwargs["peak_wheel_torque_nm"] = float(
        np.clip(anchor_nm * log_factor, _PEAK_TORQUE_NM_FLOOR, _PEAK_TORQUE_NM_CEILING)
    )
    return DesignVector(**kwargs)  # type: ignore[arg-type]


def _build_scenario_and_soil_from_lhs_row(
    family: ScenarioFamily,
    u_scenario: np.ndarray,
) -> tuple[MissionScenario, SoilParameters]:
    """Convert one scenario-perturbation LHS row to (scenario, soil).

    SCHEMA_VERSION v7_1: ``operational_duty_cycle`` is now drawn from
    the LHS uniformly over :data:`_OPERATIONAL_DUTY_CYCLE_BOUNDS`
    instead of being pinned to ``family.operational_duty_cycle``. The
    family-level default is still kept on :class:`ScenarioFamily` for
    canonical YAML / UI initial slider use.
    """
    lat_lo, lat_hi = family.latitude_range_deg
    dur_lo, dur_hi = family.mission_duration_range_days
    slope_lo, slope_hi = family.max_slope_range_deg
    duty_lo, duty_hi = _OPERATIONAL_DUTY_CYCLE_BOUNDS
    latitude = float(_unscale(u_scenario[0:1], lat_lo, lat_hi)[0])
    duration = float(_unscale(u_scenario[1:2], dur_lo, dur_hi)[0])
    max_slope = float(_unscale(u_scenario[2:3], slope_lo, slope_hi)[0])
    ops_duty = float(_unscale(u_scenario[3:4], duty_lo, duty_hi)[0])

    soil_values: dict[str, float] = {}
    for i, col in enumerate(
        [
            "soil_n",
            "soil_k_c",
            "soil_k_phi",
            "soil_cohesion_kpa",
            "soil_friction_angle_deg",
            "soil_shear_modulus_k_m",
        ],
        start=4,
    ):
        lo, hi = _SOIL_BOUNDS[col]
        soil_values[col] = float(_unscale(u_scenario[i : i + 1], lo, hi)[0])

    scenario = MissionScenario(
        name=family.name,
        latitude_deg=latitude,
        traverse_distance_m=family.traverse_distance_m,
        terrain_class=family.terrain_class,
        soil_simulant=family.soil_simulant,
        mission_duration_earth_days=duration,
        max_slope_deg=max_slope,
        sun_geometry=family.sun_geometry,
        operational_duty_cycle=ops_duty,
    )
    soil = SoilParameters(
        n=soil_values["soil_n"],
        k_c=soil_values["soil_k_c"],
        k_phi=soil_values["soil_k_phi"],
        cohesion_kpa=soil_values["soil_cohesion_kpa"],
        friction_angle_deg=soil_values["soil_friction_angle_deg"],
        shear_modulus_k_m=soil_values["soil_shear_modulus_k_m"],
    )
    return scenario, soil


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_samples(
    n_per_scenario: int,
    *,
    seed: int = 42,
    scenario_names: list[str] | None = None,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
) -> list[LHSSample]:
    """Generate a stratified LHS over the Phase-2 design space.

    Parameters
    ----------
    n_per_scenario
        Number of samples per scenario family. Must be even (so the
        50/50 n_wheels stratification is exact). Four families are used
        by default, giving ``4 * n_per_scenario`` total samples.
    seed
        Master RNG seed. All downstream randomness (per-stratum LHS,
        split assignment) is derived deterministically from this.
    scenario_names
        Optional subset of family names (keys of :data:`FAMILIES`).
        Defaults to all four canonical scenarios.
    val_frac, test_frac
        Held-out fractions. ``train_frac = 1 - val_frac - test_frac``.

    Returns
    -------
    list[LHSSample]
        Length ``n_per_scenario * len(scenario_names)``. Samples are
        ordered ``(family_0 stratum_0, family_0 stratum_1, family_1
        stratum_0, ...)``; ``sample_index`` is assigned in this order
        and is stable given the seed.
    """
    if n_per_scenario <= 0:
        raise ValueError(f"n_per_scenario must be positive (got {n_per_scenario})")
    if n_per_scenario % 2 != 0:
        raise ValueError(
            f"n_per_scenario must be even for 50/50 n_wheels stratification (got {n_per_scenario})."
        )
    names = scenario_names if scenario_names is not None else list(FAMILIES.keys())
    for name in names:
        if name not in FAMILIES:
            raise KeyError(f"unknown scenario family {name!r}. Known: {list(FAMILIES.keys())}")

    n_continuous_design = len(_CONTINUOUS_DESIGN_BOUNDS)
    n_scenario_perturb = len(_SCENARIO_PERTURB_COLS)
    n_cols = n_continuous_design + 1 + n_scenario_perturb  # +1 for grouser_count
    n_per_stratum = n_per_scenario // 2

    total = n_per_scenario * len(names)
    splits = _assign_splits(total, seed=seed, val_frac=val_frac, test_frac=test_frac)

    samples: list[LHSSample] = []
    global_idx = 0
    rng = np.random.default_rng(seed)
    for family_idx, name in enumerate(names):
        family = FAMILIES[name]
        for stratum_id, n_wheels in enumerate([4, 6]):
            # Distinct sub-seed per (family, stratum) so each gets its
            # own space-filling draw independent of the others.
            sub_seed = int(rng.integers(0, 2**31 - 1))
            u = _unit_lhs(n_per_stratum, n_cols, seed=sub_seed)
            u_continuous = u[:, :n_continuous_design]
            u_grouser = u[:, n_continuous_design]
            u_scenario = u[:, n_continuous_design + 1 :]
            for i in range(n_per_stratum):
                design = _build_design_from_lhs_row(
                    u_continuous[i], float(u_grouser[i]), n_wheels=n_wheels
                )
                scenario, soil = _build_scenario_and_soil_from_lhs_row(family, u_scenario[i])
                samples.append(
                    LHSSample(
                        sample_index=global_idx,
                        split=str(splits[global_idx]),  # type: ignore[arg-type]
                        stratum_id=stratum_id,
                        scenario_family=name,
                        design=design,
                        scenario=scenario,
                        soil=soil,
                    )
                )
                global_idx += 1
        _ = family_idx
    return samples


__all__ = [
    "FAMILIES",
    "LHSSample",
    "ScenarioFamily",
    "SplitName",
    "generate_samples",
]
