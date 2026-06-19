"""Mobility-architecture proxy for obstacle negotiation and suspension mass.

This is an *architecture-level* model, not a kinematic rocker-bogie simulation.
``mobility_architecture`` selects between a four-wheel rigid/skid-steer proxy
and a six-wheel rocker-bogie proxy. Obstacle capability scales with wheel
radius through literature-motivated step-height factors; rocker-bogie carries
an explicit suspension mass penalty in the bottom-up mass model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MobilityArchitecture = Literal["rigid_4wheel", "rocker_bogie_6wheel"]

OBSTACLE_CAPABILITY_FACTOR: dict[MobilityArchitecture, float] = {
    "rigid_4wheel": 0.5,
    "rocker_bogie_6wheel": 1.25,
}
"""Max traversable obstacle height as a fraction of wheel radius R.

Conservative proxies for conceptual design: rigid four-wheel layouts are
limited to roughly half a wheel diameter; passive rocker-bogie suspension
can negotiate obstacles on the order of one wheel radius (MER/MSL class).
"""


def wheel_count_for_architecture(architecture: MobilityArchitecture) -> int:
    """Return the drive-wheel count implied by ``architecture``."""
    return 6 if architecture == "rocker_bogie_6wheel" else 4


def architecture_for_wheel_count(n_wheels: int) -> MobilityArchitecture:
    """Map legacy ``n_wheels`` values to the closest architecture label."""
    if n_wheels == 6:
        return "rocker_bogie_6wheel"
    if n_wheels == 4:
        return "rigid_4wheel"
    raise ValueError(f"n_wheels must be 4 or 6 (got {n_wheels}).")


def obstacle_capability_m(
    architecture: MobilityArchitecture,
    wheel_radius_m: float,
) -> float:
    """Estimated max traversable obstacle height, m."""
    if wheel_radius_m <= 0.0:
        raise ValueError("wheel_radius_m must be positive.")
    return OBSTACLE_CAPABILITY_FACTOR[architecture] * wheel_radius_m


def obstacle_margin_m(
    capability_m: float,
    required_obstacle_height_m: float,
) -> float:
    """Capability minus the scenario requirement (m)."""
    return capability_m - required_obstacle_height_m


def obstacle_requirement_met(
    capability_m: float,
    required_obstacle_height_m: float,
) -> bool:
    return capability_m + 1e-12 >= required_obstacle_height_m


@dataclass(frozen=True)
class ArchitectureParams:
    """Mass penalty coefficients for the rocker-bogie proxy."""

    rocker_bogie_fixed_mass_kg: float = 0.5
    rocker_bogie_chassis_fraction: float = 0.08


def architecture_suspension_mass_kg(
    architecture: MobilityArchitecture,
    chassis_mass_kg: float,
    *,
    params: ArchitectureParams | None = None,
) -> float:
    """Suspension / linkage mass charged to rocker-bogie architectures only."""
    if architecture == "rigid_4wheel":
        return 0.0
    p = params or ArchitectureParams()
    if chassis_mass_kg <= 0.0:
        raise ValueError("chassis_mass_kg must be positive.")
    return p.rocker_bogie_fixed_mass_kg + p.rocker_bogie_chassis_fraction * chassis_mass_kg
