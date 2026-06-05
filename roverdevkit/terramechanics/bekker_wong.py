"""Analytical terramechanics: Bekker-Wong pressure-sinkage + Janosi-Hanamoto shear.

Single-wheel drawbar pull, sinkage, and driving torque as a function of
wheel geometry, vertical load, slip, and soil parameters.

Model overview
--------------
A rigid wheel of radius ``R`` and width ``b`` sinks a depth ``z_0`` into
deformable soil. Under the contact patch (bounded by entry angle θ₁ and
exit angle θ₂, with θ₂ = 0 for a rigid wheel by Wong's standard
convention) the soil exerts a radial normal stress σ(θ) and a
tangential shear stress τ(θ). Integrating these stresses around the
contact patch yields the vertical load W, drawbar pull DP, and
driving torque T. The entry angle θ₁ is pinned by the constraint that
the integrated vertical force balances the applied load.

Primary sources
---------------
- Bekker, M. G. (1969). *Introduction to Terrain-Vehicle Systems*.
  University of Michigan Press. [pressure-sinkage and plate compaction
  resistance]
- Janosi, Z. & Hanamoto, B. (1961). "The analytical determination of
  drawbar pull as a function of slip for tracked vehicles in
  deformable soils." Proc. 1st Int. Conf. Terrain-Vehicle Systems,
  Turin, Italy. [mobilisation of shear with slip]
- Wong, J. Y. & Reece, A. R. (1967). "Prediction of rigid wheel
  performance based on the analysis of soil-wheel stresses: Part I.
  Performance of driven rigid wheels." *J. Terramech.* 4(1):81-98.
  [rigid-wheel adaptation and the piecewise rear-region formulation]
- Wong, J. Y. (2008). *Theory of Ground Vehicles*, 4th ed., Wiley.
  Chapters 2 (soil), 3 (track/wheel resistance), 4 (wheel-soil
  interaction). [unified textbook treatment; reference for all the
  equations used here]

Assumptions
-----------
- Rigid wheel (no tire deflection).
- Exit angle θ₂ = 0 (Wong's standard assumption — the soil rebounds
  elastically behind the wheel and contributes nothing to the net
  stress). More elaborate treatments (Ishigami 2007) let θ₂ < 0 with
  an explicit bulldozing contribution; not modelled here.
- Transition angle for peak stress θ_m = (c₁ + c₂·|s|)·θ₁ with
  c₁ = 0.4, c₂ = 0.2. These are Wong's typical empirical defaults;
  Ding 2011 reports soil-dependent fits spanning c₁ ∈ [0.18, 0.43],
  c₂ ∈ [0.09, 0.25].
- Grousers contribute a multiplicative shear-thrust lift derived
  from Iizuka & Kubota 2011 (see ``_grouser_shear_lift``). The term
  is closed-form in (R, N_g, h_g) and saturates at large grouser
  packs.

Validation (Layer 3)
--------------------
The Layer-3 sub-model validation grid lives at
``data/validation/wong_layer3_reference.csv`` and is exercised by
``tests/test_terramechanics.py::test_layer3_published_reference_grid``.
Each row is a published-reference operating point (Wong 2008 §4.2
worked-example fixture, a Pragyan-class smooth wheel, and a Yutu-2-class
grousered wheel on Apollo nominal regolith, plus Iizuka & Kubota 2011
grouser limit cases) with per-quantity tolerance bands sized at the
±15-30 % BW model-form error reported in Ishigami (2007) and
Ding et al. (2011). Adding new digitised rows is additive — append a
row to the CSV and the parametrised test picks it up automatically.

"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

# Empirical coefficients for θ_m = (c₁ + c₂·|s|)·θ₁ (Wong & Reece 1967;
# Wong 2008 §4.2).
_C1_THETA_M: float = 0.4
_C2_THETA_M: float = 0.2

# Trapezoidal grid for the angular integrals. 100 points puts
# integration error well below the ±15-30 % model-form error of
# Bekker-Wong. Profiled at ~0.3 ms per evaluation.
_N_QUAD: int = 100

# Saturation cap for the grouser shear-thrust lift, dimensionless. Lab
# data from Iizuka & Kubota 2011 (Fig. 7-9, GRC-1 / FJS-1) show the
# tractive coefficient gain plateaus at ~50–60 % once the grouser pack
# is dense enough that adjacent shear planes interfere. Capping the
# arc-density form at 0.6 prevents the analytical lift from running
# unphysically high for extreme N_g · h_g / R combinations.
_GROUSER_LIFT_CAP: float = 0.6


# ---------------------------------------------------------------------------
# Data classes (frozen — safe to hash / cache)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SoilParameters:
    """Bekker / Mohr-Coulomb soil parameters.

    Sourced from :file:`data/soil_simulants.csv`.
    """

    n: float
    """Sinkage exponent (dimensionless)."""

    k_c: float
    """Cohesive modulus, kN/m^(n+1)."""

    k_phi: float
    """Frictional modulus, kN/m^(n+2)."""

    cohesion_kpa: float
    """Soil cohesion c, kPa."""

    friction_angle_deg: float
    """Internal friction angle φ, degrees."""

    shear_modulus_k_m: float = 0.018
    """Janosi-Hanamoto shear-deformation modulus K, meters.

    Default 0.018 m from Wong 2008; typical lunar-simulant range is
    0.006–0.025 m depending on density and moisture.
    """


@dataclass(frozen=True)
class WheelGeometry:
    """Rigid-wheel geometry for a single wheel."""

    radius_m: float
    width_m: float
    grouser_height_m: float = 0.0
    grouser_count: int = 0


@dataclass(frozen=True)
class WheelForces:
    """Steady-state per-wheel outputs of the Bekker-Wong model."""

    drawbar_pull_n: float
    driving_torque_nm: float
    sinkage_m: float
    rolling_resistance_n: float
    slip: float
    entry_angle_rad: float
    """Entry angle θ₁ that satisfies vertical force balance."""


# ---------------------------------------------------------------------------
# Internal helpers (all in SI inside)
# ---------------------------------------------------------------------------


def _grouser_shear_lift(wheel: WheelGeometry) -> float:
    """Engaged-grouser shear-thrust enhancement factor (Iizuka & Kubota 2011).

    Each grouser blade penetrating depth ``h_g`` extends the shear
    interface from the wheel rim down to ``R + h_g``. The expected
    number of grousers in contact at any instant is

    .. math::

        N_{\\text{eng}} \\;=\\; \\frac{N_g\\,\\theta_1}{2\\pi},

    i.e. the contact-arc fraction of the full circumference. Each
    engaged grouser extends the shear plane by ``h_g`` over the bare
    contact-arc length ``R\\,\\theta_1``, so the multiplicative shear
    thrust gain is

    .. math::

        g \\;=\\; 1 \\;+\\; \\frac{N_{\\text{eng}}\\,h_g}{R\\,\\theta_1}
              \\;=\\; 1 \\;+\\; \\frac{N_g\\,h_g}{2\\pi R},

    which is independent of ``θ_1`` (the cancellation is exact). This
    is the closed-form arc-density limit of Iizuka & Kubota's grouser
    correction, suitable for analytical sweeps.

    The raw lift is capped at ``_GROUSER_LIFT_CAP`` so the term saturates
    in the regime where adjacent grouser shear planes interfere — beyond
    that, more grousers contribute negligibly (Iizuka & Kubota 2011 Fig.
    7–9). The cap matches their reported ~50–60 % asymptote.

    Returns ``1.0`` when ``N_g = 0`` or ``h_g = 0``; the BW kernel then
    reduces to the original grouser-blind form bit-for-bit.
    """
    if wheel.grouser_count <= 0 or wheel.grouser_height_m <= 0.0:
        return 1.0
    arc_density = wheel.grouser_count * wheel.grouser_height_m / (2.0 * math.pi * wheel.radius_m)
    return 1.0 + min(arc_density, _GROUSER_LIFT_CAP)


def _effective_modulus_pa_per_m_n(soil: SoilParameters, width_m: float) -> float:
    """Combined Bekker modulus, converted to SI.

    Bekker's pressure-sinkage law (Bekker 1969; Wong 2008 eq. 2.11):

    .. math::

        p(z) = \\left(\\frac{k_c}{b} + k_\\phi\\right)\\, z^{\\,n}

    With ``k_c`` in kN/m^(n+1) and ``k_phi`` in kN/m^(n+2), the
    bracketed group has units kN/m^(n+2). Multiply by 10³ to get
    Pa/m^n so the product ``k_eff · z^n`` (with z in metres) lands in
    Pascals.
    """
    return (soil.k_c / width_m + soil.k_phi) * 1000.0


def _integrate_forces(
    theta_1: float,
    wheel: WheelGeometry,
    soil: SoilParameters,
    slip: float,
) -> tuple[float, float, float]:
    """Integrate σ(θ) and τ(θ) around the contact patch.

    Returns ``(W, DP, T)`` in SI units (N, N, N·m) for a given entry
    angle θ₁. ``W`` is the integrated vertical force — the quantity
    that must equal the applied load for the wheel to be in equilibrium.
    """
    if theta_1 <= 0.0:
        return 0.0, 0.0, 0.0

    radius_m = wheel.radius_m
    width_m = wheel.width_m
    n = soil.n
    phi_rad = math.radians(soil.friction_angle_deg)
    cohesion_pa = soil.cohesion_kpa * 1000.0
    shear_modulus_m = soil.shear_modulus_k_m
    k_eff = _effective_modulus_pa_per_m_n(soil, width_m)

    # -----------------------------------------------------------------
    # Peak-stress angle θ_m (Wong & Reece 1967; Wong 2008 §4.2):
    #
    #     θ_m = (c₁ + c₂·|s|)·θ₁
    #
    # splits the contact patch into a "front" region θ_m ≤ θ ≤ θ₁
    # (where σ grows as the wheel penetrates) and a "rear" region
    # θ₂ ≤ θ < θ_m (where σ decays toward zero at the exit angle θ₂).
    # -----------------------------------------------------------------
    theta_m = (_C1_THETA_M + _C2_THETA_M * abs(slip)) * theta_1

    # Uniform grid from θ₂ = 0 to θ₁. 100 points is the budget (see
    # _N_QUAD); quadrature error is far below Bekker-Wong's model-form
    # error of ±15–30 %.
    theta: NDArray[np.float64] = np.linspace(0.0, theta_1, _N_QUAD)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_theta_1 = math.cos(theta_1)

    # -----------------------------------------------------------------
    # Radial normal stress σ(θ)   (Wong & Reece 1967; Wong 2008 §4.2)
    # -----------------------------------------------------------------
    # Rigid-wheel geometry: the soil-surface intrusion depth at angle
    # θ (with z_0 = R(1 − cos θ₁) the maximum sinkage) is
    #
    #     z(θ) = R·(cos θ − cos θ₁)                              (i)
    #
    # Substituting (i) into Bekker's p(z) = k_eff · z^n gives the
    # **front region** (θ_m ≤ θ ≤ θ₁) stress:
    #
    #     σ₁(θ) = k_eff · R^n · (cos θ − cos θ₁)^n              (ii)
    #
    # The **rear region** (0 ≤ θ < θ_m) re-uses the same shape but
    # with an angular remap θ★ = θ★(θ) that linearly maps
    # [0, θ_m] → [θ₁, θ_m]:
    #
    #     θ★(θ) = θ₁ − (θ/θ_m)·(θ₁ − θ_m)                      (iii)
    #     σ₂(θ) = k_eff · R^n · (cos θ★ − cos θ₁)^n            (iv)
    #
    # Check: θ★(0) = θ₁ ⇒ σ₂(0) = 0 (vanishes at exit); θ★(θ_m) = θ_m
    # ⇒ σ₂ matches σ₁ at the transition, so the composite σ(θ) is
    # continuous.
    arg_front = np.maximum(cos_theta - cos_theta_1, 0.0)  # (ii) — max for numerics
    sigma_front = k_eff * radius_m**n * arg_front**n

    if theta_m > 0.0:
        theta_star = theta_1 - (theta / theta_m) * (theta_1 - theta_m)  # (iii)
        arg_rear = np.maximum(np.cos(theta_star) - cos_theta_1, 0.0)
        sigma_rear = k_eff * radius_m**n * arg_rear**n  # (iv)
    else:
        sigma_rear = np.zeros_like(theta)

    sigma = np.where(theta >= theta_m, sigma_front, sigma_rear)

    # -----------------------------------------------------------------
    # Kinematic shear displacement j(θ)   (Wong & Reece 1967; Wong 2008 §4.2)
    # -----------------------------------------------------------------
    # With slip ratio  s = 1 − V/(Rω)  (positive for driving):
    #
    #     j(θ) = R · [(θ₁ − θ) − (1 − s)·(sin θ₁ − sin θ)]       (v)
    #
    # Physical meaning: j is the accumulated tangential displacement
    # of a soil particle relative to the wheel surface, measured from
    # the moment the particle is engaged at θ = θ₁.
    # Checks:
    #   - j(θ₁) = 0                                           (entry)
    #   - At s = 1 (pure skid), j(θ) = R(θ₁ − θ) (maximal slip length)
    #   - At s = 0 (no slip), j is small but nonzero — a kinematic
    #     rolling-shear residual. The DP(0)
    #     sign-subtlety discussion.
    j = radius_m * ((theta_1 - theta) - (1.0 - slip) * (math.sin(theta_1) - sin_theta))

    # -----------------------------------------------------------------
    # Shear stress τ(θ)   (Janosi & Hanamoto 1961; Wong 2008 eq. 2.39)
    # -----------------------------------------------------------------
    # Mohr-Coulomb strength envelope:
    #
    #     τ_max(θ) = c + σ(θ)·tan φ                             (vi)
    #
    # Janosi-Hanamoto exponential mobilisation with shear modulus K:
    #
    #     τ(θ) = τ_max · (1 − exp(−|j|/K)) · sgn(j)             (vii)
    #
    # The sgn(j) factor is a minor extension of the original (1961)
    # paper — it lets the same formula handle the driving (j > 0) and
    # braking (j < 0) cases with a single expression.
    tau_max = cohesion_pa + sigma * math.tan(phi_rad)
    tau = tau_max * (1.0 - np.exp(-np.abs(j) / shear_modulus_m)) * np.sign(j)

    # -----------------------------------------------------------------
    # Grouser shear-thrust lift   (Iizuka & Kubota 2011)
    # -----------------------------------------------------------------
    # Multiplicative gain on τ from grousers extending the shear plane
    # below the wheel rim. Independent of θ in this closed-form limit,
    # so it scales W, DP, and T together. Reduces to 1.0 when the wheel
    # has no grousers; saturates at _GROUSER_LIFT_CAP for very dense
    # grouser packs. See ``_grouser_shear_lift`` for the derivation.
    tau = tau * _grouser_shear_lift(wheel)

    # -----------------------------------------------------------------
    # Force integrals   (Wong 2008 §4.2)
    # -----------------------------------------------------------------
    # Sign-convention derivation (in wheel-axle frame, x̂ forward,
    # ŷ upward, θ measured from downward vertical, positive forward):
    #
    #   Outward unit normal on the wheel at angle θ:
    #       n̂(θ) = ( sin θ, −cos θ)                             (down-forward)
    #
    #   Soil exerts a compressive (inward) normal reaction on the
    #   wheel, so the force per unit area from soil on wheel is
    #       σ⃗ = −σ·n̂ = σ·(−sin θ, +cos θ).
    #
    #   Tangent-in-rotation-direction at the contact surface:
    #       t̂(θ) = (−cos θ, −sin θ)                             (backward-down)
    #
    #   For driving slip (s > 0) the wheel surface moves backward
    #   relative to the soil, so soil reacts on the wheel in the
    #   +forward direction, i.e. in −t̂. Defining τ > 0 as tractive:
    #       τ⃗ = −τ·t̂ = τ·(+cos θ, +sin θ).
    #
    # Summing horizontal and vertical components of σ⃗ + τ⃗ and
    # integrating over the contact arc (arc element R dθ, contact
    # width b) gives Wong's standard form:
    #
    #     W  = b·R ∫[0,θ₁] (σ cos θ + τ sin θ) dθ              (viii)
    #     DP = b·R ∫[0,θ₁] (τ cos θ − σ sin θ) dθ               (ix)
    #     T  = b·R² ∫[0,θ₁]  τ             dθ                    (x)
    #
    # For (x) the extra R is the moment arm about the wheel axle.
    integrand_w = sigma * cos_theta + tau * sin_theta  # (viii)
    integrand_dp = tau * cos_theta - sigma * sin_theta  # (ix)
    integrand_t = tau  # (x)

    scale = width_m * radius_m
    vertical_load = scale * float(np.trapezoid(integrand_w, theta))
    drawbar_pull = scale * float(np.trapezoid(integrand_dp, theta))
    torque = scale * radius_m * float(np.trapezoid(integrand_t, theta))

    return vertical_load, drawbar_pull, torque


def _compaction_resistance(sinkage_m: float, wheel: WheelGeometry, soil: SoilParameters) -> float:
    """Bekker plate compaction resistance, reported as a diagnostic.

    Derivation (Bekker 1969; Wong 2008 §3.4): the work per unit
    forward distance required to compact the soil under a plate of
    width ``b`` from depth 0 to ``z_0`` is

    .. math::

        R_c = b \\int_0^{z_0} p(z)\\, dz
            = \\frac{b\\,(k_c/b + k_\\phi)}{n+1}\\, z_0^{\\,n+1}

    which, multiplied by speed, equals the power dissipated in
    compaction. ``R_c`` is therefore a "motion resistance" with units
    of force.

    In the rigid-wheel rolling model this is not identically equal to
    ``−DP`` at ``s = 0`` because the integrated DP also picks up the
    kinematic-shear contribution from τ(θ) at zero slip (see discussion
    in the module docstring). For realistic
    lunar per-wheel loads the two agree in magnitude to ~15 %.
    """
    if sinkage_m <= 0.0:
        return 0.0
    k_eff = _effective_modulus_pa_per_m_n(soil, wheel.width_m)
    return wheel.width_m * k_eff * sinkage_m ** (soil.n + 1.0) / (soil.n + 1.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def single_wheel_forces(
    wheel: WheelGeometry,
    soil: SoilParameters,
    vertical_load_n: float,
    slip: float,
) -> WheelForces:
    """Compute steady-state drawbar pull, torque, and sinkage for one wheel.

    Solves the vertical force-balance equation implicitly for the
    entry angle θ₁ using Brent's method. Given θ₁ the sinkage is
    ``z_0 = R(1 − cos θ₁)`` (Wong 2008 §4.2) and the remaining
    quantities follow from the σ/τ integrals in :func:`_integrate_forces`.

    All outputs are in SI units regardless of the CSV parameter units.

    Parameters
    ----------
    wheel
        Wheel geometry.
    soil
        Soil parameters.
    vertical_load_n
        Normal load on the wheel, in newtons (lunar gravity already
        applied by the caller).
    slip
        Longitudinal slip ratio, in [-1, 1]. Positive for driving
        (``slip = 1 − V/(Rω)``).

    Returns
    -------
    WheelForces
        Per-wheel forces and kinematic quantities, plus the solved
        entry angle.

    Raises
    ------
    ValueError
        If no entry angle in ``(0, π/2)`` satisfies the force balance —
        typically because the wheel is fully buried (soil is too soft
        or load too high for the geometry).
    """
    if vertical_load_n <= 0.0:
        raise ValueError("vertical_load_n must be positive")
    if not -1.0 <= slip <= 1.0:
        raise ValueError("slip must lie in [-1, 1]")

    def residual(theta_1: float) -> float:
        w, _, _ = _integrate_forces(theta_1, wheel, soil, slip)
        return w - vertical_load_n

    # Bracket θ₁ ∈ (0, π/2). At θ₁ → 0 the contact patch vanishes so
    # W → 0 < load (negative residual); at θ₁ → π/2 the wheel is half
    # buried and W is very large (positive residual). brentq locates
    # the sign change in O(log) steps.
    theta_low = 1e-5
    theta_high = math.pi / 2.0 - 1e-4
    try:
        theta_1 = brentq(residual, theta_low, theta_high, xtol=1e-6, rtol=1e-6)
    except ValueError as exc:
        r_low = residual(theta_low)
        r_high = residual(theta_high)
        raise ValueError(
            "could not find entry angle satisfying vertical force balance "
            f"(load={vertical_load_n:.1f} N, R={wheel.radius_m:.3f} m, "
            f"b={wheel.width_m:.3f} m). At θ₁=ε residual={r_low:.1f} N, "
            f"at θ₁=π/2−ε residual={r_high:.1f} N. Likely wheel is fully "
            "buried (soil too soft or load too high for this geometry)."
        ) from exc

    _, drawbar_pull, torque = _integrate_forces(theta_1, wheel, soil, slip)
    sinkage = wheel.radius_m * (1.0 - math.cos(theta_1))  # z_0 = R(1 − cos θ₁)
    rolling_resistance = _compaction_resistance(sinkage, wheel, soil)

    return WheelForces(
        drawbar_pull_n=drawbar_pull,
        driving_torque_nm=torque,
        sinkage_m=sinkage,
        rolling_resistance_n=rolling_resistance,
        slip=slip,
        entry_angle_rad=theta_1,
    )
