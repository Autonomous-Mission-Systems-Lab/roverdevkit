"""Derived mobility capabilities computed from the Bekker-Wong model.

Right now this contains one thing: the **maximum climbable slope** for a
rover design on a given soil. This is separate from the traverse sim
because it's a static capability metric (not time-resolved) and it
populates ``MissionMetrics.slope_capability_deg`` for the canonical mission scenarios
scenario 3).

Approach
--------
For a rover traversing a slope of inclination theta, at steady speed:

    Tractive force required per wheel = m*g*sin(theta) / n_wheels
    Normal load per wheel              = m*g*cos(theta) / n_wheels

The Bekker-Wong model's ``drawbar_pull_n`` is the *net* horizontal force
a single wheel delivers beyond its own motion resistance, so DP must
balance the gradient term alone. At each candidate slope we evaluate
:func:`single_wheel_forces` at a reference high-slip point
(``max_slip``) to get the maximum available DP and look for the slope
at which available DP equals required DP.

We cap the search at 35 deg because (a) the :class:`MissionScenario`
schema allows ``max_slope_deg <= 35`` and (b) at that slope rover
stability (tip-over) starts to dominate over traction, which this model
ignores.
"""

from __future__ import annotations

import math

from scipy.optimize import brentq

from roverdevkit.terramechanics.bekker_wong import (
    SoilParameters,
    WheelGeometry,
    single_wheel_forces,
)

DEFAULT_LUNAR_GRAVITY_M_PER_S2: float = 1.625
"""Reference lunar gravity; matches :data:`MassModelParams.gravity_moon_m_per_s2`."""

DEFAULT_MAX_SLIP_FOR_CAPABILITY: float = 0.6
"""Reference high-slip operating point for max-DP (Wong 2008 §4.2)."""

SLOPE_SEARCH_UPPER_DEG: float = 35.0
"""Upper bound of the brentq search, matching the scenario schema cap."""


def _dp_balance_residual(
    slope_deg: float,
    *,
    wheel: WheelGeometry,
    soil: SoilParameters,
    total_mass_kg: float,
    n_wheels: int,
    gravity_m_per_s2: float,
    max_slip: float,
) -> float:
    """Available minus required drawbar pull per wheel (in N).

    Positive = rover can climb this slope with margin to spare;
    negative = unclimbable.
    """
    theta = math.radians(slope_deg)
    weight_n = total_mass_kg * gravity_m_per_s2
    load_per_wheel_n = weight_n * math.cos(theta) / n_wheels
    required_dp_n = weight_n * math.sin(theta) / n_wheels

    forces = single_wheel_forces(wheel, soil, load_per_wheel_n, slip=max_slip)
    return forces.drawbar_pull_n - required_dp_n


def max_climbable_slope_deg(
    wheel: WheelGeometry,
    soil: SoilParameters,
    total_mass_kg: float,
    n_wheels: int,
    *,
    gravity_m_per_s2: float = DEFAULT_LUNAR_GRAVITY_M_PER_S2,
    max_slip: float = DEFAULT_MAX_SLIP_FOR_CAPABILITY,
) -> float:
    """Largest slope (deg) this design can climb on this soil.

    Parameters
    ----------
    wheel, soil
        Bekker-Wong geometry and soil parameters.
    total_mass_kg
        Vehicle mass (from the mass model).
    n_wheels
        Number of driven wheels.
    gravity_m_per_s2
        Surface gravity, default lunar.
    max_slip
        Slip ratio at which to evaluate max available DP. 0.6 is the
        conventional choice for short-duration peak pull (Wong 2008).

    Returns
    -------
    float
        Slope in degrees, in ``[0, 35]``. Returns 35 if the rover can
        climb at least that steep (the schema cap). Returns 0 if the
        rover cannot move on flat ground.
    """
    if total_mass_kg <= 0.0 or n_wheels <= 0:
        raise ValueError("total_mass_kg and n_wheels must be positive.")

    def residual(slope_deg: float) -> float:
        return _dp_balance_residual(
            slope_deg,
            wheel=wheel,
            soil=soil,
            total_mass_kg=total_mass_kg,
            n_wheels=n_wheels,
            gravity_m_per_s2=gravity_m_per_s2,
            max_slip=max_slip,
        )

    if residual(0.0) <= 0.0:
        # Cannot move on flat ground (e.g. wheel is buried); return 0.
        return 0.0

    if residual(SLOPE_SEARCH_UPPER_DEG) >= 0.0:
        # Rover can climb at least the schema cap.
        return SLOPE_SEARCH_UPPER_DEG

    return float(brentq(residual, 0.0, SLOPE_SEARCH_UPPER_DEG, xtol=1e-3, rtol=1e-4))
