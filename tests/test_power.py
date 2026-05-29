"""Tests for the power sub-package.

Solar: physics-first-principles assertions plus a Yutu-2 noon-power
cross-check.

Battery: round-trip efficiency, SOC clamping, temperature derating, and
the SMAD-style usable-capacity validation gate ("100 Wh nominal pack
delivers ~85 Wh usable" at 20 C with the default 15 % DoD floor).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from roverdevkit.power.battery import (
    BatteryState,
    step,
    stored_energy_wh,
    temperature_derating_factor,
    usable_capacity_wh,
)
from roverdevkit.power.solar import (
    LUNAR_HOUR_ANGLE_RATE_DEG_PER_HR,
    LUNAR_SYNODIC_DAY_HOURS,
    SOLAR_CONSTANT_AU_1_W_PER_M2,
    lunar_hour_angle_deg,
    panel_power_w,
    solar_power_timeseries,
    sun_azimuth_deg,
    sun_elevation_deg,
)

# ---------------------------------------------------------------------------
# Solar geometry
# ---------------------------------------------------------------------------


class TestSunElevation:
    """Closed-form sanity checks against textbook spherical-astronomy cases."""

    def test_noon_sun_at_equator_is_zenith(self) -> None:
        assert sun_elevation_deg(latitude_deg=0.0, lunar_hour_angle_deg=0.0) == pytest.approx(
            90.0, abs=1e-9
        )

    def test_noon_sun_elevation_equals_complement_of_latitude(self) -> None:
        # delta = 0 => sin(el) = cos(phi) => el = 90 - |phi|
        for lat in (-60.0, -30.0, 10.0, 45.5, 80.0):
            expected = 90.0 - abs(lat)
            assert sun_elevation_deg(lat, lunar_hour_angle_deg=0.0) == pytest.approx(
                expected, abs=1e-9
            )

    def test_sun_at_horizon_when_hour_angle_is_90_deg_at_equator(self) -> None:
        assert sun_elevation_deg(latitude_deg=0.0, lunar_hour_angle_deg=90.0) == pytest.approx(
            0.0, abs=1e-9
        )

    def test_sun_below_horizon_at_midnight_at_equator(self) -> None:
        assert sun_elevation_deg(latitude_deg=0.0, lunar_hour_angle_deg=180.0) == pytest.approx(
            -90.0, abs=1e-9
        )

    def test_sun_elevation_is_symmetric_in_hour_angle(self) -> None:
        for lat in (-30.0, 0.0, 45.5):
            for h in (15.0, 60.0, 89.0):
                assert sun_elevation_deg(lat, +h) == pytest.approx(sun_elevation_deg(lat, -h))

    def test_pole_with_zero_declination_keeps_sun_at_horizon(self) -> None:
        # phi = +/-90, delta = 0 => sin(el) = 0 for all H. We use 89.999 to
        # avoid the cos(phi) singularity in the azimuth formula; that
        # 0.001 deg offset bounds the elevation error at ~0.001 deg.
        for h in (-180.0, -90.0, 0.0, 45.0, 180.0):
            assert sun_elevation_deg(latitude_deg=89.999, lunar_hour_angle_deg=h) == pytest.approx(
                0.0, abs=2e-3
            )


class TestSunAzimuth:
    def test_azimuth_due_south_at_noon_for_northern_latitude(self) -> None:
        # Northern hemisphere with delta=0: sun is due south at local noon.
        assert sun_azimuth_deg(latitude_deg=45.0, lunar_hour_angle_deg=0.0) == pytest.approx(
            180.0, abs=1e-6
        )

    def test_azimuth_in_valid_range(self) -> None:
        for lat in (-60.0, 0.0, 45.5):
            for h in (-179.9, -45.0, 0.0, 45.0, 179.9):
                az = sun_azimuth_deg(lat, h)
                assert 0.0 <= az < 360.0


class TestLunarHourAngle:
    def test_noon_returns_zero(self) -> None:
        assert lunar_hour_angle_deg(elapsed_hours=0.0, noon_hour=0.0) == pytest.approx(0.0)

    def test_full_synodic_day_wraps(self) -> None:
        wrapped = lunar_hour_angle_deg(elapsed_hours=LUNAR_SYNODIC_DAY_HOURS, noon_hour=0.0)
        assert abs(wrapped) < 1e-6 or abs(abs(wrapped) - 360.0) < 1e-6

    def test_quarter_day_advances_90_deg(self) -> None:
        h = lunar_hour_angle_deg(elapsed_hours=LUNAR_SYNODIC_DAY_HOURS / 4.0, noon_hour=0.0)
        assert h == pytest.approx(90.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Panel power
# ---------------------------------------------------------------------------


class TestPanelPower:
    def test_panel_power_is_zero_below_horizon(self) -> None:
        assert (
            panel_power_w(panel_area_m2=1.0, panel_efficiency=0.30, sun_elevation_deg=-10.0) == 0.0
        )
        assert panel_power_w(panel_area_m2=1.0, panel_efficiency=0.30, sun_elevation_deg=0.0) == 0.0

    def test_horizontal_panel_at_zenith_yields_full_irradiance(self) -> None:
        p = panel_power_w(panel_area_m2=1.0, panel_efficiency=0.30, sun_elevation_deg=90.0)
        assert p == pytest.approx(SOLAR_CONSTANT_AU_1_W_PER_M2 * 1.0 * 0.30, rel=1e-9)

    def test_horizontal_panel_follows_sin_elevation(self) -> None:
        # For beta = 0, P should scale as sin(elevation).
        for el in (10.0, 30.0, 45.5, 75.0):
            p = panel_power_w(panel_area_m2=1.0, panel_efficiency=0.30, sun_elevation_deg=el)
            expected = SOLAR_CONSTANT_AU_1_W_PER_M2 * 0.30 * math.sin(math.radians(el))
            assert p == pytest.approx(expected, rel=1e-9)

    def test_dust_factor_scales_linearly(self) -> None:
        clean = panel_power_w(
            panel_area_m2=1.0,
            panel_efficiency=0.30,
            sun_elevation_deg=45.0,
            dust_degradation_factor=1.0,
        )
        dusty = panel_power_w(
            panel_area_m2=1.0,
            panel_efficiency=0.30,
            sun_elevation_deg=45.0,
            dust_degradation_factor=0.7,
        )
        assert dusty == pytest.approx(0.7 * clean)

    def test_tilted_panel_aimed_at_sun_outperforms_horizontal(self) -> None:
        # At Yutu-2-like latitude, tilting the panel by (90 - el) toward the
        # sun's azimuth should recover (very nearly) the full irradiance.
        el = 44.5  # noon elevation at Yutu-2 latitude (45.5 N), delta=0
        sun_az = 180.0
        horiz = panel_power_w(
            panel_area_m2=1.0,
            panel_efficiency=0.30,
            sun_elevation_deg=el,
        )
        tilted = panel_power_w(
            panel_area_m2=1.0,
            panel_efficiency=0.30,
            sun_elevation_deg=el,
            panel_tilt_deg=90.0 - el,
            panel_azimuth_deg=sun_az,
            sun_azimuth_deg=sun_az,
        )
        # Tilted panel should be brighter, and very close to full irradiance.
        assert tilted > horiz
        assert tilted == pytest.approx(SOLAR_CONSTANT_AU_1_W_PER_M2 * 0.30, rel=1e-6)

    def test_back_illuminated_tilted_panel_does_not_go_negative(self) -> None:
        # Sun in front, panel tilted way back so the cosine flips sign.
        p = panel_power_w(
            panel_area_m2=1.0,
            panel_efficiency=0.30,
            sun_elevation_deg=20.0,
            panel_tilt_deg=80.0,
            panel_azimuth_deg=0.0,
            sun_azimuth_deg=180.0,
        )
        assert p == 0.0

    def test_invalid_efficiency_rejected(self) -> None:
        with pytest.raises(ValueError):
            panel_power_w(panel_area_m2=1.0, panel_efficiency=1.5, sun_elevation_deg=45.0)
        with pytest.raises(ValueError):
            panel_power_w(panel_area_m2=1.0, panel_efficiency=-0.1, sun_elevation_deg=45.0)

    def test_invalid_dust_factor_rejected(self) -> None:
        with pytest.raises(ValueError):
            panel_power_w(
                panel_area_m2=1.0,
                panel_efficiency=0.30,
                sun_elevation_deg=45.0,
                dust_degradation_factor=1.5,
            )


# ---------------------------------------------------------------------------
# Yutu-2 validation gate
# ---------------------------------------------------------------------------


class TestYutu2Validation:
    """Cross-check against the published Yutu-2 power-profile numbers.

    Yutu-2 specs (Di et al. 2020 *Icarus*; CNSA mission documents):
        - Selenographic latitude: ~45.5 N
        - Solar array: nominally 1.0 m^2, ~30 % cell efficiency
          (Chinese GaAs triple junction).
        - Reported in-flight noon-equivalent panel output: ~120-140 W,
          with the gap between cell-level theoretical and as-flown power
          attributable to dust deposition, harness/MPPT losses, thermal
          derating of the cells, and a several-degree array-tilt offset.

    The first sub-test confirms the *clean-sky theoretical* power matches
    the closed-form S * A * eta * sin(el) for the Yutu-2 geometry. The
    second sub-test shows that applying realistic loss factors (dust ~0.5,
    cell thermal derating ~0.85) brings the model into the published
    in-flight band - i.e. the unmodelled gap between physics and flight
    data is well-characterised by parameters the user can tune.
    """

    def test_yutu2_clean_sky_matches_closed_form(self) -> None:
        elev = sun_elevation_deg(latitude_deg=45.5, lunar_hour_angle_deg=0.0)
        p = panel_power_w(
            panel_area_m2=1.0,
            panel_efficiency=0.30,
            sun_elevation_deg=elev,
        )
        expected = SOLAR_CONSTANT_AU_1_W_PER_M2 * 1.0 * 0.30 * math.sin(math.radians(44.5))
        assert p == pytest.approx(expected, rel=1e-6)
        # Theoretical clean-sky upper bound for this geometry: ~286 W.
        assert 250.0 < p < 320.0

    def test_yutu2_with_realistic_losses_in_published_band(self) -> None:
        elev = sun_elevation_deg(latitude_deg=45.5, lunar_hour_angle_deg=0.0)
        # Dust + cell thermal derating bring the in-flight number down.
        p = panel_power_w(
            panel_area_m2=1.0,
            panel_efficiency=0.30 * 0.85,  # ~85 % thermal derate at lunar-noon array temp
            sun_elevation_deg=elev,
            dust_degradation_factor=0.55,  # accumulated regolith deposition
        )
        # Published in-flight: ~120-140 W noon-equivalent.
        assert 100.0 < p < 160.0


# ---------------------------------------------------------------------------
# Solar power timeseries
# ---------------------------------------------------------------------------


class TestSolarPowerTimeseries:
    def test_timeseries_has_expected_shape(self) -> None:
        t, p = solar_power_timeseries(
            duration_hours=LUNAR_SYNODIC_DAY_HOURS,
            dt_hours=10.0,
            latitude_deg=0.0,
            panel_area_m2=1.0,
            panel_efficiency=0.30,
        )
        assert t.shape == p.shape
        assert t[0] == 0.0
        assert t[-1] == pytest.approx(LUNAR_SYNODIC_DAY_HOURS, abs=10.0)
        # Default noon at quarter-day puts sunrise at t=0; midnight at half-day.
        midnight_idx = int(np.argmin(p))
        # Power should be zero for substantial portions (~half) of the cycle.
        assert (p == 0.0).sum() >= len(p) // 3
        # Non-zero peak should exceed S * A * eta * sin(some elevation).
        assert p.max() > 0.5 * SOLAR_CONSTANT_AU_1_W_PER_M2 * 0.30
        # Midnight should be deep in the dark portion.
        assert p[midnight_idx] == 0.0

    def test_lunar_day_period_constants_consistent(self) -> None:
        # Hour-angle rate * synodic day length = 360 deg.
        product = LUNAR_HOUR_ANGLE_RATE_DEG_PER_HR * LUNAR_SYNODIC_DAY_HOURS
        assert product == pytest.approx(360.0)


# ---------------------------------------------------------------------------
# Battery state-of-charge
# ---------------------------------------------------------------------------


def _fresh_state(soc: float = 1.0, **kwargs: float) -> BatteryState:
    defaults: dict[str, float] = {
        "capacity_wh": 100.0,
        "state_of_charge": soc,
        "temperature_c": 20.0,
    }
    defaults.update(kwargs)
    return BatteryState(**defaults)


class TestBatteryConstruction:
    def test_default_construction(self) -> None:
        s = BatteryState(capacity_wh=100.0, state_of_charge=0.8)
        assert s.capacity_wh == 100.0
        assert s.state_of_charge == 0.8
        assert s.min_state_of_charge == 0.15

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"capacity_wh": 0.0, "state_of_charge": 0.5},
            {"capacity_wh": -10.0, "state_of_charge": 0.5},
            {"capacity_wh": 100.0, "state_of_charge": -0.1},
            {"capacity_wh": 100.0, "state_of_charge": 1.5},
            {"capacity_wh": 100.0, "state_of_charge": 0.5, "charge_efficiency": 0.0},
            {"capacity_wh": 100.0, "state_of_charge": 0.5, "discharge_efficiency": 1.5},
            {"capacity_wh": 100.0, "state_of_charge": 0.5, "min_state_of_charge": -0.1},
        ],
    )
    def test_invalid_construction_rejected(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(ValueError):
            BatteryState(**kwargs)


class TestBatteryStep:
    def test_step_zero_dt_is_noop(self) -> None:
        s0 = _fresh_state(soc=0.5)
        s1 = step(s0, power_net_w=100.0, dt_s=0.0)
        assert s1.state_of_charge == s0.state_of_charge

    def test_charging_increases_soc(self) -> None:
        s0 = _fresh_state(soc=0.5)
        s1 = step(s0, power_net_w=10.0, dt_s=3600.0)  # 10 W * 1 h = 10 Wh in
        # eta_charge = 0.95 => stored 9.5 Wh in 100 Wh pack => +0.095
        assert s1.state_of_charge == pytest.approx(0.5 + 0.095, abs=1e-9)

    def test_discharging_decreases_soc(self) -> None:
        s0 = _fresh_state(soc=0.5)
        s1 = step(s0, power_net_w=-10.0, dt_s=3600.0)  # 10 W * 1 h = 10 Wh out
        # eta_discharge = 0.95 => cells must give up 10 / 0.95 ≈ 10.526 Wh
        assert s1.state_of_charge == pytest.approx(0.5 - (10.0 / 0.95) / 100.0, abs=1e-9)

    def test_soc_clamped_at_full(self) -> None:
        s0 = _fresh_state(soc=0.99)
        s1 = step(s0, power_net_w=100.0, dt_s=3600.0)
        assert s1.state_of_charge == pytest.approx(1.0)

    def test_soc_clamped_at_dod_floor(self) -> None:
        s0 = _fresh_state(soc=0.20, min_state_of_charge=0.15)
        s1 = step(s0, power_net_w=-100.0, dt_s=3600.0)
        assert s1.state_of_charge == pytest.approx(0.15)

    def test_round_trip_loses_energy(self) -> None:
        s0 = _fresh_state(soc=0.5)
        s1 = step(s0, power_net_w=10.0, dt_s=3600.0)
        s2 = step(s1, power_net_w=-10.0, dt_s=3600.0)
        # Net energy change should be negative (round-trip losses).
        assert s2.state_of_charge < s0.state_of_charge
        # Loss ≈ (1 - eta_c * eta_d) * 10 Wh consumed at the load
        # Charged 10 Wh in -> stored 9.5; discharged 10 Wh out -> drew 10/0.95 ≈ 10.53
        # Net stored change: 9.5 - 10.53 = -1.03 Wh -> -0.0103 SOC change.
        assert s2.state_of_charge == pytest.approx(0.5 + 0.095 - 10.0 / 0.95 / 100.0, abs=1e-9)

    def test_returned_state_is_independent_object(self) -> None:
        s0 = _fresh_state(soc=0.5)
        s1 = step(s0, power_net_w=10.0, dt_s=60.0)
        assert s0.state_of_charge == 0.5  # original untouched
        assert s1 is not s0

    def test_negative_dt_rejected(self) -> None:
        s0 = _fresh_state(soc=0.5)
        with pytest.raises(ValueError):
            step(s0, power_net_w=10.0, dt_s=-1.0)


class TestTemperatureDerating:
    def test_room_temperature_is_calibration_point(self) -> None:
        assert temperature_derating_factor(20.0) == pytest.approx(1.0)

    def test_cold_reduces_capacity(self) -> None:
        assert temperature_derating_factor(-20.0) < 1.0
        assert temperature_derating_factor(-40.0) < temperature_derating_factor(-20.0)

    def test_hot_reduces_capacity_modestly(self) -> None:
        f = temperature_derating_factor(60.0)
        assert 0.9 < f < 1.0

    def test_clamped_outside_table(self) -> None:
        assert temperature_derating_factor(-100.0) == pytest.approx(
            temperature_derating_factor(-40.0)
        )
        assert temperature_derating_factor(200.0) == pytest.approx(
            temperature_derating_factor(60.0)
        )

    def test_factor_in_unit_interval(self) -> None:
        for t in np.linspace(-50.0, 80.0, 50):
            f = temperature_derating_factor(float(t))
            assert 0.0 <= f <= 1.0


class TestUsableCapacity:
    def test_validation_gate_100wh_pack_at_room_temp(self) -> None:
        """SMAD-style sizing rule of thumb: 100 Wh nominal -> ~85 Wh usable
        at 20 C with the default 15 % DoD floor."""
        s = BatteryState(capacity_wh=100.0, state_of_charge=1.0)
        assert usable_capacity_wh(s) == pytest.approx(85.0, abs=1.0)

    def test_cold_reduces_usable_capacity(self) -> None:
        warm = usable_capacity_wh(BatteryState(capacity_wh=100.0, state_of_charge=1.0))
        cold = usable_capacity_wh(
            BatteryState(capacity_wh=100.0, state_of_charge=1.0, temperature_c=-20.0)
        )
        assert cold < warm

    def test_higher_dod_floor_reduces_usable_capacity(self) -> None:
        loose = usable_capacity_wh(
            BatteryState(capacity_wh=100.0, state_of_charge=1.0, min_state_of_charge=0.1)
        )
        strict = usable_capacity_wh(
            BatteryState(capacity_wh=100.0, state_of_charge=1.0, min_state_of_charge=0.4)
        )
        assert strict < loose


class TestStoredEnergy:
    def test_full_pack(self) -> None:
        assert stored_energy_wh(BatteryState(capacity_wh=100.0, state_of_charge=1.0)) == 100.0

    def test_half_pack(self) -> None:
        assert stored_energy_wh(BatteryState(capacity_wh=200.0, state_of_charge=0.5)) == 100.0
