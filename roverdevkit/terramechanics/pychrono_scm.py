"""PyChrono Soil Contact Model single-wheel driver (Path 2).

A drop-in analog of :func:`roverdevkit.terramechanics.bekker_wong.single_wheel_forces`
that runs a PyChrono SCM time-stepped simulation of a rigid cylindrical
wheel rolling on deformable terrain, and returns a :class:`WheelForces`
so the two models can be compared directly.

Rig (right-handed, Z up, X forward, Y lateral; gravity = −9.81 ẑ m/s²)::

    ground (fixed)
      │  linear-speed motor along world X  (prescribes forward velocity V)
      ▼
    cart_x    [nominal mass, carries no load]
      │  prismatic along world Z, free — lets the wheel sink under gravity
      ▼
    cart_z    [mass = (W_n / g) − m_wheel, delivers target vertical load]
      │  rotation-speed motor around world Y  (prescribes ω = V/(R·(1−s)))
      ▼
    wheel     [rigid cylinder R × b, nominal mass, has collision shape
               so SCM can detect contact]

Simulation timeline::

    t ∈ [0,            settle_time]           → motors at 0, wheel drops in
    t ∈ [settle_time,  settle_time + drive]   → motors at V, ω
    Averaging window: last (1 − average_window_skip) fraction of the drive phase

Signed forces on the wheel are pulled from ``SCMTerrain.GetContactForceBody``
at every step and averaged over the steady-state window to build the returned
``WheelForces``.

Optional high-fidelity path for wheel-soil data generation.
Heavier than the analytical path by 2-4 orders of magnitude, so used for
(a) PyChrono integration wiring validation, (b) real-rover validation calibration data for the
correction layer, (c) error-budget targeted validation runs.

Known environment quirk
-----------------------
The conda-forge ``pychrono`` 10.0 build on osx-arm64 has ``libChrono_core.dylib``
with undefined Intel-OpenMP symbols (``__kmpc_dispatch_init_4`` etc.) that
are not resolved via a ``NEEDED`` entry. We work around this by preloading
``libiomp5.dylib`` via :func:`ctypes.CDLL` **before** importing ``pychrono``.
The preload is a no-op on platforms where the shared library isn't present.
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import sys
import time
from dataclasses import dataclass

import numpy as np

from .bekker_wong import SoilParameters, WheelForces, WheelGeometry

# ---------------------------------------------------------------------------
# OpenMP preload shim (osx-arm64 conda-forge packaging workaround)
# ---------------------------------------------------------------------------


def _preload_openmp_runtime() -> None:
    """Preload ``libiomp5`` with ``RTLD_GLOBAL`` so PyChrono can resolve
    its Intel-OMP symbols at import time. No-op if not needed."""
    candidate_names = ("libiomp5.dylib", "libiomp5.so", "libiomp5md.dll")
    search_dirs = [
        os.path.join(sys.prefix, "lib"),
        os.path.join(sys.prefix, "Library", "bin"),
    ]
    for d in search_dirs:
        for name in candidate_names:
            path = os.path.join(d, name)
            if os.path.exists(path):
                with contextlib.suppress(OSError):
                    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                return


_preload_openmp_runtime()

try:
    import pychrono as chrono
    import pychrono.vehicle as veh

    _PYCHRONO_AVAILABLE = True
    _IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover
    chrono = None  # type: ignore[assignment, unused-ignore]
    veh = None  # type: ignore[assignment, unused-ignore]
    _PYCHRONO_AVAILABLE = False
    _IMPORT_ERROR = exc


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScmConfig:
    """Knobs for the SCM single-wheel driver.

    Defaults tuned for Rashid-class micro-rover wheels (R ~0.1 m) on
    lunar-simulant soils with O(10–50 N) per-wheel loads, balancing
    steady-state signal-to-noise against wall-clock cost.
    """

    time_step_s: float = 1e-3
    """Integrator step size (s). SCM is stable down to ~5e-4 s; 1e-3
    is typically fine for the quasi-static regime we care about."""

    settle_time_s: float = 0.3
    """Duration (s) before motors engage — the wheel drops under gravity
    into the soil patch to establish initial sinkage."""

    drive_time_s: float = 1.5
    """Duration (s) of prescribed V, ω motion after settling."""

    driving_velocity_m_s: float = 0.1
    """Forward translation speed V (m/s). Kept low to stay in the
    quasi-static regime where analytical Bekker-Wong is a fair comparator."""

    average_window_skip: float = 0.5
    """Fraction of the drive phase to skip before averaging (discards
    motor-startup transients). 0.5 → averages the last half."""

    terrain_mesh_res_m: float = 0.015
    """SCM mesh resolution δ (m). Rule of thumb: δ ≤ R/6. Finer costs
    quadratically more time."""

    patch_length_m: float = 1.2
    """SCM patch length along X (m). Must cover wheel_start_x +
    V × drive_time with margin on each side."""

    patch_width_m: float = 0.4
    """SCM patch width along Y (m)."""

    wheel_start_x_m: float = -0.3
    """Initial wheel X position (m) relative to patch centre."""

    wheel_clearance_m: float = 0.005
    """Initial air gap (m) between wheel bottom and undisturbed soil
    surface before gravity acts."""

    scm_elastic_k_pa_per_m: float = 2.0e8
    """SCM elastic stiffness (Pa/m). Must exceed Bekker K_φ. Controls
    recoverable elastic sinkage before plastic yield."""

    scm_damping_pa_s_per_m: float = 3.0e4
    """SCM damping (Pa·s/m). Proportional to vertical compaction speed."""

    enable_bulldozing: bool = False
    """Leave bulldozing off for apples-to-apples comparison with the
    analytical Bekker-Wong model, which has no bulldozing term."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def is_available() -> bool:
    """Return True iff ``import pychrono`` succeeded at module load time."""
    return _PYCHRONO_AVAILABLE


def import_error() -> Exception | None:
    """Return the exception raised while importing pychrono, or ``None``."""
    return _IMPORT_ERROR


def single_wheel_forces_scm(
    wheel: WheelGeometry,
    soil: SoilParameters,
    vertical_load_n: float,
    slip: float,
    *,
    config: ScmConfig | None = None,
    telemetry: dict[str, float] | None = None,
) -> WheelForces:
    """Run a PyChrono SCM single-wheel simulation and return time-averaged forces.

    Parameters mirror :func:`bekker_wong.single_wheel_forces`. The optional
    ``config`` argument exposes the PyChrono-specific knobs. If ``telemetry``
    is passed, this function writes ``wall_clock_s``, ``fz_mean_n``,
    ``fz_residual_n``, ``driving_velocity_m_s`` and ``target_omega_rad_s``
    into it — useful for benchmarking and for checking the vertical-force
    balance (Fz_mean should ≈ vertical_load_n in equilibrium).
    """
    if not _PYCHRONO_AVAILABLE:
        raise RuntimeError(
            f"PyChrono is not available in this environment ({_IMPORT_ERROR}). "
            "Install via `mamba env update -f environment.yml` or use the "
            "analytical `single_wheel_forces`."
        )
    if vertical_load_n <= 0.0:
        raise ValueError("vertical_load_n must be positive")
    if not -1.0 <= slip <= 1.0:
        raise ValueError("slip must lie in [-1, 1]")
    if slip >= 1.0:
        raise ValueError("slip must be < 1 (would require infinite ω)")

    cfg = config or ScmConfig()
    g = 9.81

    # ----- 1. system + terrain -------------------------------------------
    sys_ = chrono.ChSystemSMC()
    sys_.SetGravitationalAcceleration(chrono.ChVector3d(0.0, 0.0, -g))
    sys_.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)

    # Shared contact material — SCM only uses its existence, not the tuning.
    mat = chrono.ChContactMaterialSMC()

    terrain = veh.SCMTerrain(sys_)
    # SetSoilParameters is in SI: kN → N via ×1000.
    terrain.SetSoilParameters(
        soil.k_phi * 1000.0,
        soil.k_c * 1000.0,
        soil.n,
        soil.cohesion_kpa * 1000.0,
        soil.friction_angle_deg,
        soil.shear_modulus_k_m,
        cfg.scm_elastic_k_pa_per_m,
        cfg.scm_damping_pa_s_per_m,
    )
    if cfg.enable_bulldozing:
        terrain.EnableBulldozing(True)
    # Flat square patch centred at origin, top surface at Z = 0.
    terrain.Initialize(cfg.patch_length_m, cfg.patch_width_m, cfg.terrain_mesh_res_m)

    # ----- 2. bodies ------------------------------------------------------
    # Distribute the target vertical load between cart_z and wheel_body.
    # Lump most of the load into cart_z so the wheel's rotational inertia
    # stays small → motor torque is dominated by soil reaction, not
    # accelerating the wheel itself.
    wheel_mass = max(0.1, vertical_load_n / g * 0.05)
    cart_z_mass = vertical_load_n / g - wheel_mass

    ground = chrono.ChBody()
    ground.SetFixed(True)
    sys_.AddBody(ground)

    cart_x = chrono.ChBody()
    cart_x.SetMass(0.2)
    cart_x.SetPos(
        chrono.ChVector3d(cfg.wheel_start_x_m, 0.0, wheel.radius_m + cfg.wheel_clearance_m)
    )
    sys_.AddBody(cart_x)

    cart_z = chrono.ChBody()
    cart_z.SetMass(cart_z_mass)
    iner = 0.1 * cart_z_mass
    cart_z.SetInertiaXX(chrono.ChVector3d(iner, iner, iner))
    cart_z.SetPos(
        chrono.ChVector3d(cfg.wheel_start_x_m, 0.0, wheel.radius_m + cfg.wheel_clearance_m)
    )
    sys_.AddBody(cart_z)

    # Rigid wheel: cylinder with axis along local Y. Collision is enabled
    # so SCM's ray-casting contact detector can find the wheel surface.
    wheel_body = chrono.ChBodyEasyCylinder(
        chrono.ChAxis_Y,
        wheel.radius_m,
        wheel.width_m,
        1000.0,  # density — overridden below via SetMass
        True,  # create_visualization
        True,  # create_collision
        mat,
    )
    wheel_body.SetMass(wheel_mass)
    ixx = 0.5 * wheel_mass * wheel.radius_m**2
    iyy = wheel_mass * (3.0 * wheel.radius_m**2 + wheel.width_m**2) / 12.0
    wheel_body.SetInertiaXX(chrono.ChVector3d(iyy, ixx, iyy))
    wheel_body.SetPos(
        chrono.ChVector3d(cfg.wheel_start_x_m, 0.0, wheel.radius_m + cfg.wheel_clearance_m)
    )
    wheel_body.EnableCollision(True)
    sys_.AddBody(wheel_body)

    # ----- 3. joints + motors --------------------------------------------
    # Chrono convention: ChLinkMotorLinearSpeed, ChLinkLockPrismatic, and
    # ChLinkMotorRotationSpeed all use the joint frame's **Z-axis** as
    # their primary axis. Verified empirically with the identity-frame test
    # as a smoke test for the optional PyChrono integration.
    #
    # x_motor: drive along world +X → rotate local Z to world X via
    #          QuatFromAngleY(+π/2), since R_y(π/2)·ẑ = x̂.
    x_motor = chrono.ChLinkMotorLinearSpeed()
    x_motor.Initialize(
        cart_x,
        ground,
        chrono.ChFramed(
            chrono.ChVector3d(cfg.wheel_start_x_m, 0.0, wheel.radius_m),
            chrono.QuatFromAngleY(chrono.CH_PI_2),
        ),
    )
    x_speed_fn = chrono.ChFunctionConst(0.0)
    x_motor.SetSpeedFunction(x_speed_fn)
    sys_.AddLink(x_motor)

    # ChLinkLockPrismatic uses the joint frame's **Z-axis** as its sliding
    # axis. Identity quaternion → sliding along world Z.
    z_prismatic = chrono.ChLinkLockPrismatic()
    z_prismatic.Initialize(
        cart_z,
        cart_x,
        chrono.ChFramed(
            chrono.ChVector3d(cfg.wheel_start_x_m, 0.0, wheel.radius_m + cfg.wheel_clearance_m),
            chrono.ChQuaterniond(1, 0, 0, 0),
        ),
    )
    sys_.AddLink(z_prismatic)

    # ChLinkMotorRotation uses the joint frame's **Z-axis** as its rotation
    # axis. To rotate the wheel around world Y we rotate the motor frame
    # so its local Z points along world Y: QuatFromAngleX(−π/2).
    rot_motor = chrono.ChLinkMotorRotationSpeed()
    rot_motor.Initialize(
        wheel_body,
        cart_z,
        chrono.ChFramed(
            chrono.ChVector3d(cfg.wheel_start_x_m, 0.0, wheel.radius_m + cfg.wheel_clearance_m),
            chrono.QuatFromAngleX(-chrono.CH_PI_2),
        ),
    )
    rot_speed_fn = chrono.ChFunctionConst(0.0)
    rot_motor.SetSpeedFunction(rot_speed_fn)
    sys_.AddLink(rot_motor)

    # ----- 4. SCM active domain (big perf win) ---------------------------
    # Box following the wheel: 3R forward-aft × 2b lateral × 1R vertical.
    # SCM only tesselates + updates mesh nodes inside this box each step.
    terrain.AddActiveDomain(
        wheel_body,
        chrono.ChVector3d(0.0, 0.0, 0.0),
        chrono.ChVector3d(3.0 * wheel.radius_m, 2.0 * wheel.width_m, wheel.radius_m),
    )

    # ----- 5. Time stepping ---------------------------------------------
    target_v = cfg.driving_velocity_m_s
    target_omega = target_v / (wheel.radius_m * (1.0 - slip))

    # Sign of the rotation-motor input:
    #   With ω around world +Y, a wheel at height R has its bottom at
    #   r_c = (0, 0, −R). The surface velocity at the contact point is
    #     v_contact = v_center + ω × r_c
    #                = V x̂ + (ω ĵ) × (−R k̂)
    #                = V x̂ − ω R (ĵ × k̂)
    #                = V x̂ − ω R x̂
    #                = (V − ω R) x̂
    #   No-slip → V − ω R = 0 → ω = V/R (positive for forward rolling).
    #   The sign is confirmed empirically against the analytical model.
    motor_omega_sign = +1.0

    n_settle = int(round(cfg.settle_time_s / cfg.time_step_s))
    n_drive = int(round(cfg.drive_time_s / cfg.time_step_s))
    n_skip = int(round(cfg.average_window_skip * n_drive))

    fx_samples: list[float] = []
    fz_samples: list[float] = []
    ty_samples: list[float] = []
    sinkage_samples: list[float] = []

    frc = chrono.ChVector3d(0.0, 0.0, 0.0)
    trq = chrono.ChVector3d(0.0, 0.0, 0.0)

    wall_t0 = time.perf_counter()

    # Settle phase: motors held at 0, wheel drops under gravity.
    for _ in range(n_settle):
        sys_.DoStepDynamics(cfg.time_step_s)

    # Engage motors.
    x_speed_fn.SetConstant(target_v)
    rot_speed_fn.SetConstant(motor_omega_sign * target_omega)

    # Drive phase; record over the averaging window.
    for step_idx in range(n_drive):
        sys_.DoStepDynamics(cfg.time_step_s)
        if step_idx >= n_skip:
            terrain.GetContactForceBody(wheel_body, frc, trq)
            fx_samples.append(frc.x)
            fz_samples.append(frc.z)
            ty_samples.append(trq.y)
            # Sinkage = R − wheel-centre-height (soil top is at Z=0).
            sinkage_samples.append(wheel.radius_m - wheel_body.GetPos().z)

    wall_elapsed_s = time.perf_counter() - wall_t0

    if not fx_samples:
        raise RuntimeError("no averaging samples recorded; check average_window_skip")

    # ----- 6. Post-process into a WheelForces ---------------------------
    fx_mean = float(np.mean(fx_samples))
    fz_mean = float(np.mean(fz_samples))
    ty_mean = float(np.mean(ty_samples))
    sinkage_mean = float(np.mean(sinkage_samples))

    # Sign conventions (matched to bekker_wong.single_wheel_forces):
    #   drawbar_pull_n  : net horizontal soil force on the wheel in +X.
    #                     Positive for a traction-producing driving slip.
    #   driving_torque_nm: moment the motor must supply to sustain ω.
    #                      Soil exerts a reaction torque opposing rotation
    #                      (−Y for ω > 0 around +Y), so motor_T = −trq_y.
    #   sinkage_m       : positive downward (soil top − wheel bottom).
    drawbar_pull_n = fx_mean
    driving_torque_nm = -motor_omega_sign * ty_mean
    sinkage_m = max(sinkage_mean, 0.0)

    # Bekker plate-compaction resistance, reported for diagnostic parity
    # with the analytical WheelForces output (SCM itself doesn't separate
    # compaction from shear).
    from .bekker_wong import _compaction_resistance

    rolling_resistance_n = _compaction_resistance(sinkage_m, wheel, soil)

    # Infer entry angle from the SCM sinkage via the rigid-wheel geometry
    # z₀ = R(1 − cos θ₁).
    if sinkage_m < wheel.radius_m:
        entry_angle_rad = float(np.arccos(1.0 - sinkage_m / wheel.radius_m))
    else:
        entry_angle_rad = float(np.pi / 2.0)

    if telemetry is not None:
        telemetry["wall_clock_s"] = wall_elapsed_s
        telemetry["fz_mean_n"] = fz_mean
        telemetry["fz_residual_n"] = fz_mean - vertical_load_n
        telemetry["driving_velocity_m_s"] = target_v
        telemetry["target_omega_rad_s"] = target_omega
        telemetry["n_samples"] = float(len(fx_samples))

    return WheelForces(
        drawbar_pull_n=drawbar_pull_n,
        driving_torque_nm=driving_torque_nm,
        sinkage_m=sinkage_m,
        rolling_resistance_n=rolling_resistance_n,
        slip=slip,
        entry_angle_rad=entry_angle_rad,
    )
