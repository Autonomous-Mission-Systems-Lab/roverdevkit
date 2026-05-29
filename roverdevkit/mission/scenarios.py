"""Mission scenarios bundled with RoverDevKit.

Three categories of scenario co-exist in :data:`SCENARIO_DIR`:

1. **Canonical tradespace scenarios** (4) — returned by
   :func:`list_scenarios`, used by the webapp Pareto explorer, the
   surrogate-training LHS, and Layer-2 cross-scenario validation.
   These are the four "design exploration targets" sized so the
   range objective stays informative across the surrogate's LHS sweep:

   1. ``equatorial_mare_traverse`` — Apollo-17-like terrain, 14-day mission.
   2. ``polar_prospecting`` — high latitude, long shadows, intermittent sun.
   3. ``highland_slope_capability`` — up to 25° slopes, minimum-mass climber.
   4. ``crater_rim_survey`` — short traverse, lots of slope changes, energy-optimal.

2. **Class-generic micro-rover scenarios** (4, ``*_micro``) — returned
   by :func:`list_class_generic_micro_scenarios`, used **only** by the
   Layer-5 rediscovery harness
   (:mod:`roverdevkit.validation.rover_rediscovery`).
   Parallel to the canonical four (same terrain class, soil, sun
   geometry, non-binding traverse distance) but with the per-scenario
   ``operational_duty_cycle`` pinned to a flat class-neutral 0.10
   across all four — instead of the canonical 0.05 / 0.30 / 0.15 /
   0.20 anchors, which were inspection-calibrated against real-rover
   ops history (Pragyan / Apollo-17 LRV / MER, etc.). For rediscovery
   that calibration *is* the label the optimiser is asked to recover,
   so reusing the canonical YAMLs would let one observable's anchor
   propagate into the search target.
   See the per-YAML header comments for the full leakage rationale.

3. **Per-rover validation scenarios** — bespoke YAMLs for each
   registry entry (``chandrayaan3_pragyan``,
   ``change4_yutu2_per_lunar_day``, ``moonranger_polar_demo``,
   ``rashid_atlas_crater``, ``ispace_m2_tenacious``,
   ``cadre_polar_unit``). Used by
   :mod:`roverdevkit.validation.rover_comparison` for Layer-0 truth
   comparison and Layer-1 surrogate sanity; never used by rediscovery.

Scenarios are serialized as YAML in :file:`roverdevkit/mission/configs/*.yaml`
to make them easy for users and reviewers to inspect. The loader validates
via the :class:`MissionScenario` pydantic model so invalid fields raise
immediately at load time rather than deep inside the traverse sim.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import yaml  # type: ignore[import-untyped]

from roverdevkit.schema import MissionScenario, ScenarioName

SCENARIO_DIR: Path = Path(__file__).parent / "configs"

_CANONICAL_NAMES: set[str] = {
    "equatorial_mare_traverse",
    "polar_prospecting",
    "highland_slope_capability",
    "crater_rim_survey",
}
"""Canonical tradespace scenarios that :func:`list_scenarios` returns.

Other scenario categories (class-generic micro-rover via
``*_micro.yaml``; per-rover validation YAMLs) also live in
:data:`SCENARIO_DIR` but are excluded from the tradespace listing so
webapp sweeps never accidentally pick them up."""


_CLASS_GENERIC_MICRO_NAMES: set[str] = {
    "polar_micro",
    "mare_micro",
    "highland_micro",
    "crater_rim_micro",
}
"""Class-generic micro-rover scenarios used by the Layer-5
rediscovery harness in :mod:`roverdevkit.validation.rover_rediscovery`.

Parallel to :data:`_CANONICAL_NAMES` but with
``operational_duty_cycle`` pinned to a flat class-neutral 0.10 (vs.
the canonical scenarios' inspection-calibrated values) so no
real-rover ops anchor leaks into the rediscovery search target. See
the per-YAML header comments and the rediscovery module docstring
for the full leakage rationale.
"""


def _config_path(name: str) -> Path:
    return SCENARIO_DIR / f"{name}.yaml"


def load_scenario(name: str) -> MissionScenario:
    """Load a named canonical scenario from its YAML config.

    Parameters
    ----------
    name
        Scenario key; must match a ``*.yaml`` basename in
        :data:`SCENARIO_DIR`. The ``ScenarioName`` type alias pins the
        allowed values at the type-check level.

    Raises
    ------
    FileNotFoundError
        If no YAML file exists for ``name``.
    pydantic.ValidationError
        If the YAML contents do not validate against
        :class:`MissionScenario` (e.g. out-of-range latitude).
    """
    path = _config_path(name)
    if not path.exists():
        available = list_scenarios()
        raise FileNotFoundError(
            f"scenario config {path} not found. Available scenarios: {available}"
        )
    with path.open() as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(
            f"scenario file {path} did not parse to a mapping (got {type(raw).__name__})."
        )
    return MissionScenario(**raw)


def list_scenarios() -> list[ScenarioName]:
    """List the canonical tradespace scenarios that ship with the package.

    Validation-only scenarios (e.g. ``chandrayaan3_pragyan``) and
    class-generic micro-rover scenarios (``*_micro``) are kept out of
    this list so webapp sweeps never pick them up. Returned as a list
    of :data:`ScenarioName` literals; every element is guaranteed
    loadable by :func:`load_scenario`.

    See :func:`list_class_generic_micro_scenarios` for the parallel
    library used by the Layer-5 rediscovery harness.
    """
    on_disk = {p.stem for p in SCENARIO_DIR.glob("*.yaml")}
    return cast(
        "list[ScenarioName]",
        sorted(on_disk & _CANONICAL_NAMES),
    )


def list_class_generic_micro_scenarios() -> list[str]:
    """List the class-generic micro-rover scenarios bundled with the package.

    These scenarios are parallel to the four canonical tradespace
    scenarios (same terrain class / soil / sun geometry / traverse-
    distance non-binding budget) but pin ``operational_duty_cycle``
    to a flat class-neutral 0.10 across all four, breaking the
    inspection-calibration that the canonical scenarios carry against
    real-rover ops history.

    Returned in alphabetical order. Every name is guaranteed loadable
    by :func:`load_scenario`. Used **only** by
    :mod:`roverdevkit.validation.rover_rediscovery`; webapp sweeps and
    LHS training do not see this list.
    """
    on_disk = {p.stem for p in SCENARIO_DIR.glob("*.yaml")}
    missing = _CLASS_GENERIC_MICRO_NAMES - on_disk
    if missing:
        raise FileNotFoundError(
            "class-generic micro-rover scenario YAMLs missing from "
            f"{SCENARIO_DIR}: {sorted(missing)}"
        )
    return sorted(_CLASS_GENERIC_MICRO_NAMES)
