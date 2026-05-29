"""Lunar solar geometry and panel power model.

Computes instantaneous power generation given latitude, time of lunar day,
panel area, tilt, efficiency, and a dust-degradation factor.

Scope and fidelity
------------------
This is a *tradespace-level* model. We deliberately use closed-form
spherical-astronomy expressions and a constant solar irradiance instead of
SPICE/JPL ephemeris look-ups: for the design-variable
sweeps and surrogate-training runs in this project, a few-percent error in
mean daily insolation is well below the uncertainty introduced by the
mass-model and terramechanics fits.

Sign and frame conventions
--------------------------
- Latitude:  positive = lunar north; range [-90, +90] deg.
- Hour angle: 0 deg = local lunar noon; positive = afternoon; range [-180, +180].
- Sun elevation: angle above the local horizontal plane; range [-90, +90].
  Negative = below the horizon (night).
- Sun azimuth: measured clockwise from local north, in [0, 360).
- Panel tilt: 0 deg = horizontal (collector facing zenith); positive tilt
  rotates the surface normal toward the panel azimuth.
- Panel azimuth: same convention as the sun azimuth.

References
----------
Larson, W. J. & Wertz, J. R. *Space Mission Analysis and Design (SMAD)*,
3rd ed., Microcosm/Springer, 1999. Ch. 11 (electrical power), App. F
(astronomical/celestial geometry).

Patel, M. R. *Spacecraft Power Systems*, 2nd ed., CRC Press, 2017.
Ch. 4 (solar array design), Ch. 5 (sun-pointing geometry).

Heiken, G., Vaniman, D. & French, B. M. (eds.) *Lunar Sourcebook*,
Cambridge University Press, 1991. Ch. 3 (lunar environment, including
the synodic vs sidereal day distinction and the Moon's small obliquity).

Validation
-------------------
Cross-check noon power for Yutu-2 (45.5 deg N selenographic latitude)
against published Yutu-2 power-profile numbers; see ``tests/test_power.py``.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Physical / astronomical constants
# ---------------------------------------------------------------------------

SOLAR_CONSTANT_AU_1_W_PER_M2: float = 1361.0
"""Total solar irradiance at 1 AU (CODATA / NASA SORCE-era value), W/m^2.

For the Earth-Moon system this varies by about +/-3.4 % over the year
because of Earth orbital eccentricity; we treat it as a constant since the
mean is what matters for tradespace-scale integrals.
"""

LUNAR_SYNODIC_DAY_HOURS: float = 29.530589 * 24.0
"""Mean synodic (sun-to-sun) lunar day in Earth hours, ~708.73 h.

This is the period that determines local solar time on the Moon and is
therefore the relevant cycle for power-system sizing, not the 27.32-day
sidereal period.
"""

LUNAR_HOUR_ANGLE_RATE_DEG_PER_HR: float = 360.0 / LUNAR_SYNODIC_DAY_HOURS
"""Apparent rate of solar motion across the lunar sky, ~0.508 deg/hr."""

LUNAR_OBLIQUITY_DEG: float = 1.5424
"""Inclination of the lunar equator to the ecliptic, deg.

Because this is so small, the solar declination as seen from the Moon
stays within +/-1.5 deg year-round. We expose declination as a parameter
for completeness but default it to zero in higher-level helpers.
"""


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def lunar_hour_angle_deg(elapsed_hours: float, noon_hour: float = 0.0) -> float:
    """Solar hour angle on the Moon, wrapped to [-180, +180] deg.

    Parameters
    ----------
    elapsed_hours
        Wall-clock time since the start of the simulation, in Earth hours.
    noon_hour
        Time of local lunar noon (the moment the sun crosses the meridian)
        in the same time base.

    Returns
    -------
    float
        Hour angle in degrees: 0 at noon, +90 at sunset, +/-180 at midnight.
    """
    h = (elapsed_hours - noon_hour) * LUNAR_HOUR_ANGLE_RATE_DEG_PER_HR
    return ((h + 180.0) % 360.0) - 180.0


def sun_elevation_deg(
    latitude_deg: float,
    lunar_hour_angle_deg: float,
    declination_deg: float = 0.0,
) -> float:
    """Sun elevation above the local horizontal plane.

    Standard spherical-astronomy altitude formula (SMAD App. F):

        sin(el) = sin(phi) * sin(delta) + cos(phi) * cos(delta) * cos(H)

    where phi is latitude, delta is solar declination and H is the hour angle.

    Parameters
    ----------
    latitude_deg
        Selenographic latitude, deg.
    lunar_hour_angle_deg
        Hour angle, deg (0 at noon, positive in the afternoon).
    declination_deg
        Solar declination as seen from the Moon, deg. The lunar obliquity is
        only ~1.5 deg, so 0 is a good default for tradespace work.

    Returns
    -------
    float
        Elevation in deg, in [-90, +90]. Negative means the sun is below the
        local horizon (night).
    """
    phi = math.radians(latitude_deg)
    delta = math.radians(declination_deg)
    h = math.radians(lunar_hour_angle_deg)
    sin_el = math.sin(phi) * math.sin(delta) + math.cos(phi) * math.cos(delta) * math.cos(h)
    sin_el = max(-1.0, min(1.0, sin_el))  # guard against FP overshoot
    return math.degrees(math.asin(sin_el))


def sun_azimuth_deg(
    latitude_deg: float,
    lunar_hour_angle_deg: float,
    declination_deg: float = 0.0,
) -> float:
    """Sun azimuth, measured clockwise from local north, in [0, 360) deg.

    Computed via the standard horizontal-coordinate transform:

        sin(az) = -cos(delta) * sin(H) / cos(el)
        cos(az) = (sin(delta) - sin(el) * sin(phi)) / (cos(el) * cos(phi))

    where ``phi`` is latitude (``latitude_deg``), ``delta`` is solar
    declination (``declination_deg``), ``H`` is the hour angle
    (``lunar_hour_angle_deg``), ``el`` is the sun elevation derived from
    the altitude formula in :func:`sun_elevation_deg`, and ``az`` is the
    azimuth returned by this function.

    The sign convention puts az=0 at local north, az=90 at east, az=180 at
    south and az=270 at west, mirroring the standard SMAD/Patel definition.

    For points within ~0.1 deg of the geographic pole, ``cos(phi)`` is
    numerically singular and we return 0.0 by convention; tradespace runs
    that pin polar latitudes should explicitly set ``latitude_deg = +/-89.9``.
    """
    phi = math.radians(latitude_deg)
    delta = math.radians(declination_deg)
    h = math.radians(lunar_hour_angle_deg)

    sin_el = math.sin(phi) * math.sin(delta) + math.cos(phi) * math.cos(delta) * math.cos(h)
    sin_el = max(-1.0, min(1.0, sin_el))
    cos_el = math.sqrt(max(0.0, 1.0 - sin_el * sin_el))
    if cos_el < 1e-9 or abs(math.cos(phi)) < 1e-9:
        return 0.0

    sin_az = -math.cos(delta) * math.sin(h) / cos_el
    cos_az = (math.sin(delta) - sin_el * math.sin(phi)) / (cos_el * math.cos(phi))
    return math.degrees(math.atan2(sin_az, cos_az)) % 360.0


# ---------------------------------------------------------------------------
# Panel power
# ---------------------------------------------------------------------------


def panel_power_w(
    panel_area_m2: float,
    panel_efficiency: float,
    sun_elevation_deg: float,
    panel_tilt_deg: float = 0.0,
    panel_azimuth_deg: float = 180.0,
    sun_azimuth_deg: float = 180.0,
    dust_degradation_factor: float = 1.0,
    solar_constant_w_per_m2: float = SOLAR_CONSTANT_AU_1_W_PER_M2,
) -> float:
    """Instantaneous DC electrical power from a flat-plate solar array.

    The collector receives irradiance ``S * cos(i)`` where ``i`` is the angle
    between the sun line and the panel surface normal. For a panel tilted by
    ``beta`` toward azimuth ``psi``:

        cos(i) = sin(el) * cos(beta) + cos(el) * sin(beta) * cos(az_sun - psi)

    (Patel, *Spacecraft Power Systems*, eq. 5.6). For a horizontal panel
    (``panel_tilt_deg = 0``) this collapses to ``cos(i) = sin(el)``.

    Output power is then

        P = S * A * eta * max(0, cos(i)) * dust_factor.

    Symbol key (math -> Python parameter / module constant):

        P           output electrical power, W                (return value)
        S           top-of-atmosphere solar irradiance, W/m^2 (``solar_constant_w_per_m2``)
        A           active collector area, m^2                (``panel_area_m2``)
        eta         DC conversion efficiency, in [0, 1]       (``panel_efficiency``)
        i           sun-to-panel-normal incidence angle, deg  (derived)
        el          sun elevation above local horizontal, deg (``sun_elevation_deg``)
        az_sun      sun azimuth clockwise from north, deg     (``sun_azimuth_deg``)
        beta        panel tilt off horizontal, deg in [0, 90] (``panel_tilt_deg``)
        psi         panel azimuth clockwise from north, deg   (``panel_azimuth_deg``)
        dust_factor optical dust degradation, in [0, 1]       (``dust_degradation_factor``)

    Returns 0 W when the sun is at or below the horizon (night). ``cos(i)``
    is also clamped at 0 so a back-illuminated panel does not produce
    negative power.

    Parameters
    ----------
    panel_area_m2
        Active collector area, m^2.
    panel_efficiency
        DC conversion efficiency (cell + harness + MPPT) as a fraction in
        [0, 1].
    sun_elevation_deg
        Sun elevation above the local horizontal plane, deg.
    panel_tilt_deg
        Panel tilt angle off horizontal, deg in [0, 90].
    panel_azimuth_deg
        Direction the tilted panel faces (clockwise from north), deg.
        Ignored when ``panel_tilt_deg == 0``.
    sun_azimuth_deg
        Sun azimuth (clockwise from north), deg. Required only for tilted
        panels; for horizontal panels the cosine of incidence depends only
        on elevation.
    dust_degradation_factor
        Multiplicative degradation in [0, 1]; 1.0 = clean panel.
    solar_constant_w_per_m2
        Top-of-atmosphere solar irradiance, W/m^2. Defaults to the 1-AU
        value; pass a different number to model an Earth-orbit perturbation.

    Returns
    -------
    float
        Electrical power output, W (>= 0).
    """
    if panel_area_m2 < 0.0:
        raise ValueError("panel_area_m2 must be non-negative.")
    if not 0.0 <= panel_efficiency <= 1.0:
        raise ValueError("panel_efficiency must lie in [0, 1].")
    if not 0.0 <= dust_degradation_factor <= 1.0:
        raise ValueError("dust_degradation_factor must lie in [0, 1].")
    if not 0.0 <= panel_tilt_deg <= 90.0:
        raise ValueError("panel_tilt_deg must lie in [0, 90].")

    if sun_elevation_deg <= 0.0:
        return 0.0

    el = math.radians(sun_elevation_deg)
    beta = math.radians(panel_tilt_deg)
    daz = math.radians(sun_azimuth_deg - panel_azimuth_deg)

    cos_incidence = math.sin(el) * math.cos(beta) + math.cos(el) * math.sin(beta) * math.cos(daz)
    cos_incidence = max(0.0, cos_incidence)

    return (
        solar_constant_w_per_m2
        * panel_area_m2
        * panel_efficiency
        * cos_incidence
        * dust_degradation_factor
    )


# ---------------------------------------------------------------------------
# Diurnal time series helper
# ---------------------------------------------------------------------------


def solar_power_timeseries(
    duration_hours: float,
    dt_hours: float,
    latitude_deg: float,
    panel_area_m2: float,
    panel_efficiency: float,
    *,
    declination_deg: float = 0.0,
    noon_hour: float = LUNAR_SYNODIC_DAY_HOURS / 4.0,
    panel_tilt_deg: float = 0.0,
    panel_azimuth_deg: float = 180.0,
    dust_degradation_factor: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate a power-vs-time profile over one or more lunar diurnal cycles.

    Useful for plotting (see paper/paper_figures.ipynb) and as a
    reference implementation against which the traverse simulator can be
    sanity-checked.

    The default ``noon_hour`` places sunrise at t=0, so the first quarter of
    the synodic day climbs from horizon to zenith. Override it to align with
    a specific mission start condition.

    Returns
    -------
    times_hours, power_w : numpy arrays
        Same length, equally spaced by ``dt_hours``; both inclusive of the
        end of the integration window (``duration_hours``).
    """
    if duration_hours <= 0.0 or dt_hours <= 0.0:
        raise ValueError("duration_hours and dt_hours must be positive.")

    n_steps = int(math.floor(duration_hours / dt_hours)) + 1
    times = np.linspace(0.0, dt_hours * (n_steps - 1), n_steps)
    powers = np.empty_like(times)

    for i, t in enumerate(times):
        h_angle = lunar_hour_angle_deg(float(t), noon_hour=noon_hour)
        elev = sun_elevation_deg(latitude_deg, h_angle, declination_deg=declination_deg)
        if panel_tilt_deg == 0.0:
            sun_az = 180.0  # unused for horizontal panel
        else:
            sun_az = sun_azimuth_deg(latitude_deg, h_angle, declination_deg=declination_deg)
        powers[i] = panel_power_w(
            panel_area_m2=panel_area_m2,
            panel_efficiency=panel_efficiency,
            sun_elevation_deg=elev,
            panel_tilt_deg=panel_tilt_deg,
            panel_azimuth_deg=panel_azimuth_deg,
            sun_azimuth_deg=sun_az,
            dust_degradation_factor=dust_degradation_factor,
        )
    return times, powers
