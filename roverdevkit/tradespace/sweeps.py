"""Parametric sweep engine (1-D or 2-D).

User fixes a base design and a scenario, picks one (or two) design-vector
variables to sweep on a grid, and the sweep engine returns the chosen
target metric over the grid. The webapp's ``/sweep`` route is the
canonical caller; the same API is also useful from notebooks for
reproducing paper figures.

API
---

- :class:`SweepAxis` — variable name + lo/hi/n_points (linspace).
- :class:`SweepSpec`  — target metric + x axis + optional y axis + backend.
- :class:`SweepResult` — x/y grid + values + which backend ran.
- :func:`expand_grid` — cartesian product of axes applied as overrides
  on a base :class:`DesignVector`.
- :func:`pick_backend` — small auto-vs-explicit dispatcher with size guards.

The actual physics dispatch (corrected evaluator vs quantile XGB
surrogate) lives in :mod:`webapp.backend.services.sweep` because it
needs the loaded artifact handles. Keeping this module pure-Python +
numpy means it stays trivially testable without joblib / xgboost
imports at test collection time.

Why 1-D / 2-D only
------------------
The tradespace UI uses Plotly line charts (1-D) and heatmaps (2-D);
3-D sweeps are unwieldy to render and are better expressed as
NSGA-II Pareto fronts (Pareto-front workflow). If a future step needs N-D
sweeps for offline batch generation, ``itertools.product`` over a
list of :class:`SweepAxis` would be a one-line extension.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from roverdevkit.schema import DesignVector
from roverdevkit.surrogate.features import PRIMARY_REGRESSION_TARGETS

# ---------------------------------------------------------------------------
# Sweepable design variables
# ---------------------------------------------------------------------------

SWEEPABLE_VARIABLES: tuple[str, ...] = (
    "wheel_radius_m",
    "wheel_width_m",
    "grouser_height_m",
    "grouser_count",
    "chassis_mass_kg",
    "wheelbase_m",
    "solar_area_m2",
    "battery_capacity_wh",
    "avionics_power_w",
    "peak_wheel_torque_nm",
)
"""Design-vector fields the UI lets the user sweep on a grid axis.

SCHEMA_VERSION v6 (v6 schema update): ``nominal_speed_mps`` is gone (cruise
speed is now derived inside the evaluator; sweeping it is no longer
meaningful) and ``drive_duty_cycle`` was renamed ``designed_duty_cycle``.
``peak_wheel_torque_nm`` is the new drivetrain-capability sweep axis.
SCHEMA_VERSION v7 (v7 schema follow-up) drops
``designed_duty_cycle`` from the design vector entirely; drive duty
cycle is now a per-scenario quantity (``operational_duty_cycle``) and
sweeping it lives on the scenario side of the ``/sweep`` route.

``n_wheels`` is excluded because it is binary {4, 6}; a "sweep" with
two cells is better expressed by toggling it in the design panel.
``grouser_count`` is integer 0-24 but linearly spaced sweeps round to
the nearest integer; the engine handles that in :func:`expand_grid`.
"""

INTEGER_VARIABLES: frozenset[str] = frozenset({"grouser_count"})
"""Subset of :data:`SWEEPABLE_VARIABLES` whose grid values are rounded
to the nearest int before becoming :class:`DesignVector` overrides.
Pydantic on ``DesignVector`` validates the field as ``int`` so the
schema would otherwise reject a fractional grid point."""


# ---------------------------------------------------------------------------
# Result + spec containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SweepAxis:
    """One axis of a parametric sweep.

    Linearly spaced grid from ``lo`` to ``hi`` with ``n_points`` cells
    (inclusive at both ends). Bounds are not enforced here -- the
    webapp validates that they sit inside the schema range so that an
    out-of-bounds ``DesignVector`` build raises a 422 at the HTTP
    boundary instead of producing garbage results.
    """

    variable: str
    lo: float
    hi: float
    n_points: int

    def values(self) -> np.ndarray:
        """Return the grid as a 1-D numpy array, length ``n_points``."""
        if self.n_points < 2:
            raise ValueError(
                f"SweepAxis.n_points must be >= 2 for {self.variable!r} "
                f"(got {self.n_points})."
            )
        if self.hi <= self.lo:
            raise ValueError(
                f"SweepAxis.hi must be > lo for {self.variable!r} "
                f"(got lo={self.lo}, hi={self.hi})."
            )
        return np.linspace(self.lo, self.hi, self.n_points)


@dataclass(frozen=True)
class SweepSpec:
    """1-D or 2-D parametric sweep definition.

    A 2-D sweep produces a (n_y, n_x) grid; the convention matches
    Plotly's heatmap orientation (rows = y / outer index, cols = x /
    inner index), so a result can be passed directly to ``go.Heatmap``
    without transposition.
    """

    target: str
    x_axis: SweepAxis
    y_axis: SweepAxis | None
    backend: str = "auto"
    """One of ``"auto"``, ``"evaluator"``, ``"surrogate"``. The
    canonical strings are also pinned in the webapp's Pydantic schema."""

    def __post_init__(self) -> None:
        if self.target not in PRIMARY_REGRESSION_TARGETS:
            raise ValueError(
                f"target {self.target!r} is not a primary regression target "
                f"(allowed: {PRIMARY_REGRESSION_TARGETS})."
            )
        if self.x_axis.variable not in SWEEPABLE_VARIABLES:
            raise ValueError(
                f"x_axis variable {self.x_axis.variable!r} is not sweepable "
                f"(allowed: {SWEEPABLE_VARIABLES})."
            )
        if self.y_axis is not None:
            if self.y_axis.variable not in SWEEPABLE_VARIABLES:
                raise ValueError(
                    f"y_axis variable {self.y_axis.variable!r} is not sweepable "
                    f"(allowed: {SWEEPABLE_VARIABLES})."
                )
            if self.y_axis.variable == self.x_axis.variable:
                raise ValueError(
                    f"x and y axes must sweep different variables "
                    f"(both are {self.x_axis.variable!r})."
                )
        if self.backend not in {"auto", "evaluator", "surrogate"}:
            raise ValueError(
                f"backend {self.backend!r} not in {{'auto', 'evaluator', 'surrogate'}}."
            )

    def n_cells(self) -> int:
        """Total grid size (1-D returns ``x_axis.n_points``)."""
        if self.y_axis is None:
            return self.x_axis.n_points
        return self.x_axis.n_points * self.y_axis.n_points


@dataclass(frozen=True)
class SweepResult:
    """Sweep output: grid axes + value matrix + provenance."""

    spec: SweepSpec
    x_values: np.ndarray
    """1-D, length ``spec.x_axis.n_points``."""

    y_values: np.ndarray | None
    """1-D, length ``spec.y_axis.n_points``; ``None`` for 1-D sweeps."""

    z_values: np.ndarray
    """1-D ``(n_x,)`` for 1-D sweeps; 2-D ``(n_y, n_x)`` for 2-D."""

    backend_used: str
    """Concrete backend that ran (``"evaluator"`` or ``"surrogate"``)."""

    elapsed_s: float


@dataclass(frozen=True)
class SweepSensitivity:
    """Per-axis spread of the swept metric, used for UI sensitivity hints.

    Computed once on the server from a finished sweep. The frontend
    consumes ``axis_spread`` to decide whether to surface a hint like
    "y-axis only contributes 1 / 10th the spread of x" or
    "metric is saturated on this grid".

    Conventions
    -----------
    - ``total_spread`` = ``z.max() - z.min()``.
    - ``axis_spread[x]`` = median over y of ``z[:, j].max() - z[:, j].min()``
      (1-D sweeps simply use the total spread for the one axis).
    - ``axis_spread[y]`` = median over x of ``z[j, :].max() - z[j, :].min()``.
    - ``relative_spread`` = ``total_spread / max(|max|, |min|, ε)``.
      Dimensionless; small values flag a near-uniform output.
    """

    total_spread: float
    relative_spread: float
    axis_spread_x: float
    axis_spread_y: float | None


def compute_sensitivity(result: SweepResult) -> SweepSensitivity:
    """Marginal-spread sensitivity over the finished sweep grid.

    See :class:`SweepSensitivity` for the exact convention. Returns
    finite values for any non-empty grid; the relative_spread of a
    grid where all values are zero is reported as 0.0.
    """
    z = np.asarray(result.z_values, dtype=float)
    z_finite = z[np.isfinite(z)]
    if z_finite.size == 0:
        return SweepSensitivity(
            total_spread=0.0,
            relative_spread=0.0,
            axis_spread_x=0.0,
            axis_spread_y=None if result.y_values is None else 0.0,
        )

    z_max = float(np.nanmax(z))
    z_min = float(np.nanmin(z))
    total = z_max - z_min
    scale = max(abs(z_max), abs(z_min), 1e-12)
    rel = total / scale

    if result.y_values is None or z.ndim == 1:
        return SweepSensitivity(
            total_spread=total,
            relative_spread=rel,
            axis_spread_x=total,
            axis_spread_y=None,
        )

    # 2-D: median marginal spread along each axis. Median (rather than
    # max) damps a single anomalous row from drowning out the rest of
    # the surface; mean would over-weight outliers in the other
    # direction. Median is the robust split.
    #
    # z has shape (n_y, n_x): rows = y, cols = x.
    # Spread along x at fixed y_j is z[j, :].max() - z[j, :].min(),
    # so np.nanmax(z, axis=1) - np.nanmin(z, axis=1) is the "x spread"
    # per y-row. Median over rows gives the typical x spread.
    spread_along_x = np.nanmax(z, axis=1) - np.nanmin(z, axis=1)  # length n_y
    spread_along_y = np.nanmax(z, axis=0) - np.nanmin(z, axis=0)  # length n_x
    return SweepSensitivity(
        total_spread=total,
        relative_spread=rel,
        axis_spread_x=float(np.nanmedian(spread_along_x)),
        axis_spread_y=float(np.nanmedian(spread_along_y)),
    )


# ---------------------------------------------------------------------------
# Grid expansion + backend selection
# ---------------------------------------------------------------------------


def _override_design(base: DesignVector, **overrides: float | int) -> DesignVector:
    """Return ``base`` with the given fields replaced.

    Centralises the round-and-cast for integer variables so the caller
    can pass a numpy float and still get a valid Pydantic build.
    """
    payload = base.model_dump()
    for name, value in overrides.items():
        if name in INTEGER_VARIABLES:
            payload[name] = int(round(float(value)))
        else:
            payload[name] = float(value)
    return DesignVector(**payload)


def expand_grid(spec: SweepSpec, base_design: DesignVector) -> list[DesignVector]:
    """Cartesian product of x (and y) axis values applied as overrides.

    Returns
    -------
    list[DesignVector]
        For 1-D sweeps: ``[D(x0), D(x1), …, D(xN-1)]`` (length ``n_x``).
        For 2-D sweeps: row-major (y outer, x inner), so cell
        ``(j, i)`` in the result matrix is ``flat[j*n_x + i]``.

    Raises
    ------
    pydantic.ValidationError
        If a grid point falls outside the :class:`DesignVector` bounds.
        The webapp Pydantic layer normally catches this earlier; the
        raise here is the last line of defence.
    """
    xs = spec.x_axis.values()
    if spec.y_axis is None:
        return [
            _override_design(base_design, **{spec.x_axis.variable: x}) for x in xs
        ]
    ys = spec.y_axis.values()
    out: list[DesignVector] = []
    for y in ys:
        for x in xs:
            out.append(
                _override_design(
                    base_design,
                    **{spec.x_axis.variable: x, spec.y_axis.variable: y},
                )
            )
    return out


# Cell-count thresholds. Tuned for ~40 ms / cell on the corrected
# evaluator (BW/SCM bake-off) and effectively zero per cell on the vectorised
# surrogate. The auto threshold targets a sub-10 s response for
# interactive UX; the hard limits keep a malicious or sloppy request
# from melting the server.
EVALUATOR_AUTO_THRESHOLD: int = 200
EVALUATOR_HARD_LIMIT: int = 2500
SURROGATE_HARD_LIMIT: int = 40_000


def pick_backend(spec: SweepSpec) -> str:
    """Resolve ``spec.backend`` into a concrete ``"evaluator"`` or ``"surrogate"``.

    Auto-mode picks the evaluator below
    :data:`EVALUATOR_AUTO_THRESHOLD` cells and the surrogate above.
    Explicit modes are honoured but the per-backend hard limits still
    apply -- callers must not exceed
    :data:`EVALUATOR_HARD_LIMIT` / :data:`SURROGATE_HARD_LIMIT`.
    """
    n = spec.n_cells()
    if spec.backend == "auto":
        chosen = "evaluator" if n <= EVALUATOR_AUTO_THRESHOLD else "surrogate"
    else:
        chosen = spec.backend
    if chosen == "evaluator" and n > EVALUATOR_HARD_LIMIT:
        raise ValueError(
            f"sweep with {n} cells exceeds the evaluator hard limit "
            f"({EVALUATOR_HARD_LIMIT}). Reduce resolution or pick "
            "backend='surrogate'."
        )
    if chosen == "surrogate" and n > SURROGATE_HARD_LIMIT:
        raise ValueError(
            f"sweep with {n} cells exceeds the surrogate hard limit "
            f"({SURROGATE_HARD_LIMIT}). Reduce resolution."
        )
    return chosen


__all__ = [
    "EVALUATOR_AUTO_THRESHOLD",
    "EVALUATOR_HARD_LIMIT",
    "INTEGER_VARIABLES",
    "SURROGATE_HARD_LIMIT",
    "SWEEPABLE_VARIABLES",
    "SweepAxis",
    "SweepResult",
    "SweepSensitivity",
    "SweepSpec",
    "compute_sensitivity",
    "expand_grid",
    "pick_backend",
]
