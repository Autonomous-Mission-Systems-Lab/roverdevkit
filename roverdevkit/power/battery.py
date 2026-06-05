"""Battery state-of-charge model with efficiency and temperature derating.

Coulomb-counting SOC update with:
    - separate charge/discharge round-trip efficiencies,
    - depth-of-discharge floor (don't drain below ``min_state_of_charge``),
    - upper SOC clamp at 1.0 (no over-charge),
    - simple piecewise-linear temperature derating of usable capacity.

Calibration / references
------------------------
Smart, M. C. et al. *Lithium-ion electrolytes for low-temperature
operation of NASA missions*. JPL/NASA reports (multiple 2003-2018);
summarised in Halpert & Surampudi (NASA Glenn) battery technology
overviews. The piecewise-linear capacity vs temperature curve below is a
deliberately coarse fit to those datasets - good enough for tradespace
sizing where battery thermal control keeps cells within ~5-30 C - and
flagged here so it can be swapped for a vendor-specific curve later.

Larson & Wertz, *SMAD* 3rd ed., Ch. 11, gives the conventional
``E_usable = E_nominal * (1 - DoD_floor) * eta_round_trip`` accounting
that we follow.

Validation
----------
Confirm that the default model (``min_state_of_charge = 0.15``,
``charge_efficiency = discharge_efficiency = 0.95``, T = 20 C) returns
~85 Wh usable for a 100 Wh nominal pack; see ``tests/test_power.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

# ---------------------------------------------------------------------------
# Temperature derating
# ---------------------------------------------------------------------------

# Piecewise-linear approximation of Li-ion usable-capacity fraction vs cell
# temperature. Anchors are coarse but bracket the relevant operating range
# for thermally-managed lunar micro-rover packs (Smart et al.; Halpert &
# Surampudi). A spline fit would be no more accurate than this given the
# vendor-to-vendor scatter.
_TEMP_DERATING_TEMPS_C: tuple[float, ...] = (-40.0, -20.0, 0.0, 20.0, 60.0)
_TEMP_DERATING_FACTORS: tuple[float, ...] = (0.50, 0.70, 0.85, 1.00, 0.95)


def temperature_derating_factor(temperature_c: float) -> float:
    """Usable-capacity multiplier vs temperature, dimensionless.

    Returns 1.0 at 20 C (calibration point) and drops at both cold and hot
    extremes. Clamped to the endpoints outside the tabulated range so the
    model never extrapolates to non-physical multipliers (negative
    capacity, or capacity > 1).
    """
    return float(
        np.interp(
            temperature_c,
            _TEMP_DERATING_TEMPS_C,
            _TEMP_DERATING_FACTORS,
            left=_TEMP_DERATING_FACTORS[0],
            right=_TEMP_DERATING_FACTORS[-1],
        )
    )


# ---------------------------------------------------------------------------
# State container
# ---------------------------------------------------------------------------


@dataclass
class BatteryState:
    """Mutable battery state advanced each traverse-sim step."""

    capacity_wh: float
    """Nominal capacity at 20 C and full DoD, Wh."""

    state_of_charge: float
    """Fraction in [0, 1] of the *nominal* capacity stored."""

    temperature_c: float = 20.0
    """Cell temperature, deg C. Held constant within a step."""

    min_state_of_charge: float = 0.15
    """Depth-of-discharge floor; ``step`` will not drain below this.

    0.15 reflects a moderate-cycle-life DoD for orbital-grade Li-ion. Set
    higher for very-long-life applications, or 0.0 to allow full
    discharge in stress-case studies.
    """

    charge_efficiency: float = 0.95
    """Coulombic+conversion efficiency from solar bus to stored energy."""

    discharge_efficiency: float = 0.95
    """Coulombic+conversion efficiency from stored energy to load."""

    def __post_init__(self) -> None:
        if self.capacity_wh <= 0.0:
            raise ValueError("capacity_wh must be positive.")
        if not 0.0 <= self.state_of_charge <= 1.0:
            raise ValueError("state_of_charge must lie in [0, 1].")
        if not 0.0 <= self.min_state_of_charge <= 1.0:
            raise ValueError("min_state_of_charge must lie in [0, 1].")
        if not 0.0 < self.charge_efficiency <= 1.0:
            raise ValueError("charge_efficiency must lie in (0, 1].")
        if not 0.0 < self.discharge_efficiency <= 1.0:
            raise ValueError("discharge_efficiency must lie in (0, 1].")


# ---------------------------------------------------------------------------
# Update step
# ---------------------------------------------------------------------------


def step(state: BatteryState, power_net_w: float, dt_s: float) -> BatteryState:
    """Advance the battery state by one time step.

    Parameters
    ----------
    state
        Current battery state. Not mutated; a fresh ``BatteryState`` is
        returned with the updated SOC.
    power_net_w
        Net power flowing *into* the battery over the step, W.
        Positive = net charging (solar > load), negative = net discharge
        (load > solar).
    dt_s
        Time step length, s. Must be non-negative.

    Returns
    -------
    BatteryState
        New state with ``state_of_charge`` advanced and clamped to
        ``[min_state_of_charge, 1.0]``.

    Notes
    -----
    Energy book-keeping (SMAD-style):

        if charging:    dE_stored = P_net * eta_charge * dt
        if discharging: dE_stored = P_net / eta_discharge * dt        (P_net < 0)

    The asymmetric efficiency placement is deliberate: when charging, only
    a fraction ``eta_charge`` of bus energy lands in the cells; when
    discharging, the cells must give up ``|P_load| / eta_discharge`` to
    deliver ``|P_load|`` to the load. The combined round-trip ratio is
    therefore ``eta_charge * eta_discharge``.

    The clamp at ``min_state_of_charge`` and ``1.0`` silently caps energy
    flow; a separate constraint flag in the mission evaluator
    (``MissionMetrics.energy_margin_pct``) records how often the clamp
    activates.
    """
    if dt_s < 0.0:
        raise ValueError("dt_s must be non-negative.")
    if dt_s == 0.0:
        return replace(state)

    if power_net_w >= 0.0:
        delta_energy_wh = power_net_w * state.charge_efficiency * (dt_s / 3600.0)
    else:
        delta_energy_wh = (power_net_w / state.discharge_efficiency) * (dt_s / 3600.0)

    new_soc = state.state_of_charge + delta_energy_wh / state.capacity_wh
    new_soc = max(state.min_state_of_charge, min(1.0, new_soc))
    return replace(state, state_of_charge=new_soc)


def usable_capacity_wh(state: BatteryState) -> float:
    """Energy that can actually be delivered to the load from a full charge.

    Combines three loss / margin terms:

        E_usable = C_nominal * (1 - SOC_floor) * f_T(T_cell)

    where ``f_T`` is the piecewise-linear temperature derating curve. The
    discharge efficiency is *not* folded in here because callers vary in
    how they prefer to treat it (some lump it into a load model); apply it
    explicitly if you want delivered-to-load energy.

    Returns
    -------
    float
        Usable energy from a fully charged pack, Wh.
    """
    return (
        state.capacity_wh
        * (1.0 - state.min_state_of_charge)
        * temperature_derating_factor(state.temperature_c)
    )


def stored_energy_wh(state: BatteryState) -> float:
    """Energy currently stored in the pack (without any deratings), Wh.

    Convenience accessor for the traverse simulator and notebooks.
    """
    return state.capacity_wh * state.state_of_charge
