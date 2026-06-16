"""Service layer: route handlers delegate to functions here.

Routes stay thin (parse request, call service, format response); all
business logic — feature-row construction, surrogate dispatch, future
NSGA-II orchestration — lives under this package.
"""

from __future__ import annotations

from roverdevkit.schema import MissionScenario

__all__ = ["apply_scenario_overrides"]


def apply_scenario_overrides(
    scenario: MissionScenario,
    *,
    operational_duty_cycle: float | None = None,
    payload_mass_kg: float | None = None,
    payload_power_w: float | None = None,
    mission_duration_earth_days: float | None = None,
) -> MissionScenario:
    """Return a scenario copy with any provided per-call overrides applied.

    Centralises the ``scenario.model_copy(update=...)`` block every route
    shares so the Mission-Inputs panel can override δ_ops, mission duration,
    and the schema-v9 payload requirement (``payload_mass_kg`` /
    ``payload_power_w``) without each route re-implementing the merge.
    ``None`` fields fall through to the scenario's calibrated/class-typical
    default. Returns the input scenario unchanged when no override is supplied
    (no needless copy).
    """
    update: dict[str, float] = {}
    if operational_duty_cycle is not None:
        update["operational_duty_cycle"] = operational_duty_cycle
    if payload_mass_kg is not None:
        update["payload_mass_kg"] = payload_mass_kg
    if payload_power_w is not None:
        update["payload_power_w"] = payload_power_w
    if mission_duration_earth_days is not None:
        update["mission_duration_earth_days"] = mission_duration_earth_days
    return scenario.model_copy(update=update) if update else scenario
