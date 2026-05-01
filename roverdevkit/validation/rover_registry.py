"""Published-rover design vectors and mission scenarios.

This module codifies the lunar rovers we compare the evaluator and the
surrogate against (project_plan.md §6 W5 + §6 W6) as
:class:`DesignVector` + :class:`MissionScenario` pairs, plus the
published truth numbers in ``data/published_traverse_data.csv``.

Two-tier registry
-----------------
The registry is split into two tiers via :attr:`RoverRegistryEntry.is_flown`:

- **Flown** (``is_flown=True``): rovers with actual ground-truth flight
  data. Used by:

  * Layer-0 truth comparison (Week 5 acceptance gate,
    :func:`roverdevkit.validation.rover_comparison.compare_all`),
    which scores the evaluator vs published traverse / peak-solar /
    thermal data.
  * Layer-1 surrogate sanity check (Week 6,
    :func:`roverdevkit.surrogate.baselines.predict_for_registry_rovers`).

  Currently: **Pragyan** (Chandrayaan-3, 2023), **Yutu-2** (Chang'e-4,
  2019).

- **Design-target** (``is_flown=False``): well-spec'd lunar micro-rover
  designs that did not fly (lander loss or still in development). Used
  only for Layer-1 surrogate sanity. Layer-0 truth comparison is
  skipped because there's no ground-truth flight data.

  Currently: **MoonRanger** (CMU/Astrobotic, in development),
  **Rashid-1** (MBRSC/UAE, lost on Hakuto-R Mission 1, 2023).

Helpers:

- :func:`registry`        — all entries (flown + design-target).
- :func:`flown_registry`  — flown subset (Layer-0 use).
- :func:`registry_by_name` — single lookup, all tiers.

Scope decisions
---------------
- **Sojourner removed (2026-04-25).** Was a Mars-gravity sentinel; its
  multiple OOD-ness in the surrogate's design / scenario / gravity
  space made it counterproductive for the Layer-1 sanity check.
  Project narrowed to lunar micro-rover scope.
- **Iris not added.** Battery-only rover (no solar array) violates the
  surrogate's energy-architecture assumptions; would require a schema
  extension to model honestly.
- **Not a tradespace input.** These scenarios live next to the canonical
  four in :data:`SCENARIO_DIR` but are excluded from
  :func:`list_scenarios` so Phase-3 sweeps never pick them up.

Every design-vector field that is not directly published has an entry
in :attr:`RoverRegistryEntry.imputation_notes`; these notes mirror the
pattern from the Week-3 mass validation set for consistency.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from roverdevkit.mission.scenarios import load_scenario
from roverdevkit.power.thermal import ThermalArchitecture
from roverdevkit.schema import DesignVector, MissionScenario

GRAVITY_MOON_M_PER_S2: float = 1.625

DEFAULT_TRUTH_CSV: Path = (
    Path(__file__).resolve().parents[2] / "data" / "published_traverse_data.csv"
)


# ---------------------------------------------------------------------------
# Registry entry + truth data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoverRegistryEntry:
    """One rover bundled with the scenario it actually flew (or would have).

    Attributes
    ----------
    rover_name
        Short key used to look up published truth in
        :func:`load_truth_table` (flown rovers only).
    design
        12-D design vector reconstructed from public specs + documented
        imputations.
    scenario
        Mission context matching the real rover's operating environment
        (or its design-target landing site for non-flown entries).
    gravity_m_per_s2
        Passed through to the evaluator as ``gravity_m_per_s2``. Lunar
        for all current entries (the Mars-gravity Sojourner sentinel
        was removed when the project narrowed to lunar micro-rovers).
    thermal_architecture
        Per-rover thermal model (RHU power, surface area, hibernation
        load, sink temperatures) capturing the rover's actual thermal
        design rather than a generic default.
    panel_efficiency
        DC-level conversion efficiency at the rover's operating point.
        Distinct from the tradespace-default 0.28 (GaAs triple-junction
        beginning-of-life) because real rovers use different cell techs
        and see end-of-life degradation that the default doesn't model.
    panel_dust_factor
        Mission-integrated dust-transmission factor in (0, 1].
        Rover-specific because real dust accumulation is highly
        mission-dependent; lunar day-1 values differ from steady-state.
    is_flown
        Whether the rover successfully deployed and produced
        ground-truth flight data. Drives whether the entry participates
        in the Layer-0 truth comparison (see module docstring).
    imputation_notes
        Per-field notes on which design-vector entries were imputed and
        how.
    """

    rover_name: str
    design: DesignVector
    scenario: MissionScenario
    gravity_m_per_s2: float
    thermal_architecture: ThermalArchitecture
    panel_efficiency: float
    panel_dust_factor: float
    is_flown: bool
    imputation_notes: str


@dataclass(frozen=True)
class PublishedTruth:
    """Published truth values for one rover-scenario pair (flown rovers only)."""

    rover_name: str
    scenario_name: str
    traverse_m_published: float
    traverse_m_low: float
    traverse_m_high: float
    peak_solar_power_w_published: float
    peak_solar_power_w_low: float
    peak_solar_power_w_high: float
    thermal_survival_published: bool
    mission_duration_published_days: float
    citation: str
    notes: str


# ---------------------------------------------------------------------------
# Registry builders — flown rovers
# ---------------------------------------------------------------------------


def _pragyan_entry() -> RoverRegistryEntry:
    # Published specs: 26 kg total (ISRO press kit), 6 wheels at r=85 mm,
    # ~50 W avionics during active ops, ~60 Wh battery, ~0.5 m^2
    # deployable solar array.
    # Imputations (mirrors mass_validation_set.csv row for consistency):
    # - wheel_width_m = 0.07 (scaled from n_wheels geometry);
    # - chassis_mass_kg = 10 (~38 % of total per class ROT);
    # - wheelbase_m = 0.5 (from published images);
    # - peak_wheel_torque_nm = 0.85 (v5-implicit anchor at 26 kg / 6 / R=0.085;
    #   matches the order of magnitude of Pragyan's actual hub torque
    #   given its slow-traverse / low-slope design point);
    # - grouser_height_m, grouser_count from class heritage (Rashid/
    #   Yutu-style 8 mm x 12 grousers).
    # Schema v6 (W12 step B): nominal_speed_mps no longer a design
    # input (cruise speed is derived); drive_duty_cycle renamed to
    # designed_duty_cycle. Schema v7 (W12 step B follow-up):
    # designed_duty_cycle dropped from the design vector — drive
    # duty cycle now lives only on the per-scenario YAML (Pragyan
    # ~0.008, see chandrayaan3_pragyan.yaml).
    design = DesignVector(
        wheel_radius_m=0.085,
        wheel_width_m=0.07,
        grouser_height_m=0.008,
        grouser_count=12,
        n_wheels=6,
        chassis_mass_kg=10.0,
        wheelbase_m=0.5,
        solar_area_m2=0.5,
        battery_capacity_wh=60.0,
        avionics_power_w=20.0,
        peak_wheel_torque_nm=0.85,
    )
    # Thermal: Pragyan did NOT carry RHUs and died in lunar night.
    # Default architecture (rhu_power_w=0) correctly predicts failure.
    thermal = ThermalArchitecture(
        surface_area_m2=0.25,
        rhu_power_w=0.0,
        hibernation_power_w=2.0,
    )
    return RoverRegistryEntry(
        rover_name="Pragyan",
        design=design,
        scenario=load_scenario("chandrayaan3_pragyan"),
        gravity_m_per_s2=GRAVITY_MOON_M_PER_S2,
        thermal_architecture=thermal,
        panel_efficiency=0.22,  # ISRO space-grade triple-junction, BOL
        panel_dust_factor=0.85,  # Lunar Day 1 only; limited dust build-up
        is_flown=True,
        imputation_notes=(
            "wheel_width, wheelbase, grouser_height/count, chassis_mass, "
            "peak_wheel_torque_nm imputed from class heritage and "
            "published ops. avionics_power set to 20 W (design-space "
            "floor for a 26 kg rover). v6: torque anchor 0.85 Nm from "
            "sizing_peak_torque_anchor at the published 26 kg total. "
            "v7: drive duty cycle now lives on the scenario only "
            "(designed_duty_cycle removed from the design vector)."
        ),
    )


def _yutu2_entry() -> RoverRegistryEntry:
    # Published specs (Di et al. 2020; Ding et al. 2022): 135 kg total,
    # 6 wheels at r=150 mm, wheel width ~150 mm, two-wing deployable
    # solar array ~1.3 m^2, ~130 Wh Li-ion pack, continuous drive speed
    # 40 mm/s with a drive duty cycle concentrated in a few Earth-day
    # ops window per lunar day.
    # Imputations:
    # - chassis_mass_kg = 30 (the 70 kg validation-set value bakes in
    #   the 25 kg science payload; 30 kg is the "chassis+bus" minus
    #   payload for pure-mobility modelling);
    # - wheelbase_m = 1.0 (published photos);
    # - grouser specs: h=0.012 m x 18 (Yutu-class wheels are grousered);
    # - avionics_power_w = 20 (steady-state CPU+comms+sensors; 40 W only
    #   during peak drive+heater operation, which is a different case);
    # - peak_wheel_torque_nm = 5.0 (Yutu-class 6-wheel mobility motors
    #   are sized for ~5 Nm per hub; consistent with the v5-implicit
    #   anchor at the published all-up 135 kg total mass and R=0.15).
    # Note: Yutu-2 has a published all-up flight mass of ~135 kg; the
    # registry holds chassis_mass at 35 kg because that is the published
    # chassis ex-payload value (the analytical mass-up model adds payload
    # / power-system / motor / structure margins on top). After the v3
    # LHS bounds widening (chassis ceiling 35 -> 50 kg), this 35 kg
    # value sits inside the surrogate's training support rather than at
    # the corner.
    design = DesignVector(
        wheel_radius_m=0.15,
        wheel_width_m=0.15,
        grouser_height_m=0.012,
        grouser_count=18,
        n_wheels=6,
        chassis_mass_kg=35.0,  # published chassis ex-payload
        wheelbase_m=1.0,
        solar_area_m2=1.3,
        battery_capacity_wh=130.0,
        avionics_power_w=20.0,
        peak_wheel_torque_nm=5.0,
    )
    # Thermal: Yutu-class carries Pu-238 RHUs on a thermally-controlled
    # avionics box wrapped in MLI with low-alpha/high-eps surface
    # finish (silverised OSR, alpha~0.15). The lumped-parameter thermal
    # model assumes the full surface_area_m2 radiates to the cold sink,
    # which is pessimistic for a real MLI-insulated box; we use an
    # "effective radiating area" of 0.10 m^2 to represent the MLI
    # reduction. Combined with 15 W RHU + 5 W hibernation, this gives
    # cold-case equilibrium ~-18 C and hot-case ~+40 C, both in-spec.
    thermal = ThermalArchitecture(
        surface_area_m2=0.10,
        absorptivity=0.15,
        rhu_power_w=15.0,
        hibernation_power_w=5.0,
        max_operating_temp_c=60.0,  # industrial-temp-range Chinese avionics
    )
    return RoverRegistryEntry(
        rover_name="Yutu-2",
        design=design,
        scenario=load_scenario("change4_yutu2_per_lunar_day"),
        gravity_m_per_s2=GRAVITY_MOON_M_PER_S2,
        thermal_architecture=thermal,
        panel_efficiency=0.20,  # Chinese triple-junction EOL after many
        panel_dust_factor=0.55,  # lunar days (major dust accumulation)
        is_flown=True,
        imputation_notes=(
            "chassis_mass set to 35 kg (published ex-payload chassis "
            "value; in-distribution under v3 LHS bounds 3-50 kg). "
            "Yutu-2's all-up flight mass is ~135 kg including payload, "
            "structure, and power system margins which the analytical "
            "mass-up model adds on top of chassis_mass. wheelbase, "
            "grouser specs imputed from published images and the "
            "per-lunar-day ~25 m drive distance target. v6: "
            "peak_wheel_torque_nm=5.0 from class-typical Yutu-2 hub "
            "motor sizing (~5 Nm per drive). v7: drive duty cycle "
            "now lives on the scenario only (designed_duty_cycle "
            "removed from the design vector)."
        ),
    )


# ---------------------------------------------------------------------------
# Registry builders — design-target (non-flown) rovers
# ---------------------------------------------------------------------------


def _moonranger_entry() -> RoverRegistryEntry:
    # Direct cites (Kumar et al. i-SAIRAS 2020 #5068, MoonRanger Project
    # labs page, Astrobotic NASA LSITP award):
    # - chassis_mass_kg total: 13 kg
    # - n_wheels: 4
    # - max mechanical speed: 0.07 m/s ("7 cm/sec")
    # - mission duration: 8 Earth days
    # - rover length: ~0.65 m (half-length 0.325 m used for FOV calc)
    # - camera height: 0.25 m
    # - lunar South Pole, no RHU (operates in single daylight period).
    #
    # Imputations (back-solve + class match to Rashid-1):
    # - wheel_radius_m = 0.10, wheel_width_m = 0.08: class-match to
    #   Rashid-1 (10 kg, r = 0.10 m, w = 0.08 m). MoonRanger photos on
    #   labs.ri.cmu.edu show similar wheel proportions to Rashid.
    # - grouser_height_m = 0.012, grouser_count = 12: class-typical for
    #   ~0.10 m radius lunar wheel (12 % of radius); photos show
    #   prominent grousers.
    # - wheelbase_m = 0.40: body length ~0.65 m minus wheel diameter
    #   ~ 0.45 m, rounded to 0.40.
    # - solar_area_m2 = 0.30: polar back-solve. 1 km/Earth-day at
    #   ~0.05 m/s nominal => ~5.5 h drive per day. 30 W drive + 25 W
    #   avionics x 24 h ~ 1320 Wh/day. With 8 h sun and 0.20 effective
    #   eff at low elevation => ~165 W solar peak => 0.30 m^2 array.
    # - battery_capacity_wh = 100: ~3-4 h off-sun continuous ops + dawn
    #   cold-start; class-typical for 13 kg polar rover.
    # - avionics_power_w = 25: NVIDIA TX2i (~10 W) + space-hardened RTOS
    #   MCU (~3 W) + cameras + IMU + sun sensor + comms ~ 25 W active.
    # - peak_wheel_torque_nm = 0.75: v5-implicit anchor at 13 kg / 4 /
    #   R=0.10. Slightly above the schema floor; consistent with
    #   class-typical micro-rover hub torque sizings.
    design = DesignVector(
        wheel_radius_m=0.10,
        wheel_width_m=0.08,
        grouser_height_m=0.012,
        grouser_count=12,
        n_wheels=4,
        chassis_mass_kg=13.0,
        wheelbase_m=0.40,
        solar_area_m2=0.30,
        battery_capacity_wh=100.0,
        avionics_power_w=25.0,
        peak_wheel_torque_nm=0.75,
    )
    # Thermal: MoonRanger carries no RHU (Kumar et al. 2020); operates
    # only in lunar daylight at the polar landing site. Polar thermal
    # design favours low alpha to keep hot-case rejection manageable
    # given near-continuous low-elevation sun.
    thermal = ThermalArchitecture(
        surface_area_m2=0.20,
        absorptivity=0.20,
        rhu_power_w=0.0,
        hibernation_power_w=2.0,
    )
    return RoverRegistryEntry(
        rover_name="MoonRanger",
        design=design,
        scenario=load_scenario("moonranger_polar_demo"),
        gravity_m_per_s2=GRAVITY_MOON_M_PER_S2,
        thermal_architecture=thermal,
        panel_efficiency=0.28,  # modern triple-junction GaAs BOL
        panel_dust_factor=0.95,  # brand-new array, 8-day mission
        is_flown=False,
        imputation_notes=(
            "Cited: total mass (13 kg), n_wheels (4), max mech speed "
            "(0.07 m/s), mission duration (8 d), no RHU. Imputed: wheel "
            "radius/width and grousers (class-match to Rashid-1); "
            "wheelbase from published rover length; solar / battery / "
            "avionics from a power budget back-solve against the "
            "kilometer-per-day exploration target."
        ),
    )


def _rashid1_entry() -> RoverRegistryEntry:
    # Direct cites (Hurrell et al. 2025 SSR 221:37 wheel paper,
    # Els et al. LPSC 2021 #1905 instrumentation paper, ESA + Wikipedia
    # ELM page):
    # - chassis_mass_kg total: 10 kg
    # - n_wheels: 4
    # - wheel_radius_m: 0.10 ("radius of 100 mm")
    # - wheel_width_m: 0.08 ("width of 80 mm")
    # - grouser_height_m: 0.015 (15 mm flight grouser; Hurrell 2025
    #   distinguishes from the 20 mm closed-side test wheel)
    # - grouser_count: 14
    # - wheelbase_m: 0.50 (footprint 0.535 x 0.539 m per LPSC 2021)
    # - landing site: Atlas crater, Mare Frigoris (~47 N, 44 E)
    # - mission duration: 1 lunar day (~14 Earth days), no RHU.
    # - Hurrell 2025 used 0.02 m/s as the experimental drive velocity;
    #   that is now scenario-side context rather than a design input
    #   under the v6 schema (cruise speed is derived).
    #
    # Imputations:
    # - solar_area_m2 = 0.25: 0.5 x 0.5 m chassis with deployable mast;
    #   flat array bound ~0.25 m^2. Power back-solve: at lunar noon
    #   ~ 47 N, 0.20 eff x 0.25 m^2 x 0.85 dust ~ 32 W peak, sufficient
    #   for the science-heavy ~15 W avionics with battery buffering.
    # - battery_capacity_wh = 50: class-typical for 10 kg rover with
    #   14-day target; supports overnight Wi-Fi data return to lander.
    # - avionics_power_w = 15: 2x wide-field cameras + CAM-M micro
    #   imager + CAM-T thermal imager + 4x Langmuir probes + Wi-Fi
    #   comms (Els et al. 2021 inventory).
    # - peak_wheel_torque_nm = 0.5: v5-implicit anchor at 10 kg / 4 /
    #   R=0.10 sits near the schema floor; consistent with the very
    #   low-slope, very-slow micro-rover design point.
    design = DesignVector(
        wheel_radius_m=0.10,
        wheel_width_m=0.08,
        grouser_height_m=0.015,
        grouser_count=14,
        n_wheels=4,
        chassis_mass_kg=10.0,
        wheelbase_m=0.50,
        solar_area_m2=0.25,
        battery_capacity_wh=50.0,
        avionics_power_w=15.0,
        peak_wheel_torque_nm=0.5,
    )
    # Thermal: Rashid-1 carries no RHU. Mid-latitude diurnal swing
    # benefits from balanced absorptivity; the actual flight rover used
    # MLI + heaters but we don't model the latter explicitly.
    thermal = ThermalArchitecture(
        surface_area_m2=0.18,
        absorptivity=0.30,
        rhu_power_w=0.0,
        hibernation_power_w=2.0,
    )
    return RoverRegistryEntry(
        rover_name="Rashid-1",
        design=design,
        scenario=load_scenario("rashid_atlas_crater"),
        gravity_m_per_s2=GRAVITY_MOON_M_PER_S2,
        thermal_architecture=thermal,
        panel_efficiency=0.28,  # modern triple-junction GaAs BOL
        panel_dust_factor=0.85,  # Lunar Day 1 only (matches Pragyan)
        is_flown=False,
        imputation_notes=(
            "Cited (Hurrell et al. 2025 SSR; Els et al. LPSC 2021): "
            "total mass, n_wheels, wheel radius/width, grouser height "
            "(flight 15 mm) and count (14), wheelbase. Imputed: solar "
            "/ battery / avionics from a power-budget back-solve "
            "against the science-payload inventory and single-lunar-"
            "day mission target. v6: peak_wheel_torque_nm=0.5 from "
            "v5-implicit hub-torque anchor at the published 10 kg / "
            "4-wheel / R=0.10 m design point."
        ),
    )


# ---------------------------------------------------------------------------
# Registry accessors
# ---------------------------------------------------------------------------


def registry() -> tuple[RoverRegistryEntry, ...]:
    """Return the frozen tuple of all registry entries (flown + design-target).

    Use this for Layer-1 surrogate sanity checks (Week 6+). For Layer-0
    truth comparisons, use :func:`flown_registry` instead.
    """
    return (
        _pragyan_entry(),
        _yutu2_entry(),
        _moonranger_entry(),
        _rashid1_entry(),
    )


def flown_registry() -> tuple[RoverRegistryEntry, ...]:
    """Return only the rovers that successfully deployed and flew.

    Used by the Week-5 acceptance gate
    (:func:`roverdevkit.validation.rover_comparison.compare_all`)
    because design-target rovers have no published flight truth.
    """
    return tuple(e for e in registry() if e.is_flown)


def registry_by_name(name: str) -> RoverRegistryEntry:
    """Look up a single registry entry by rover name (any tier)."""
    for entry in registry():
        if entry.rover_name == name:
            return entry
    raise KeyError(f"unknown rover {name!r}; registry has {[e.rover_name for e in registry()]}.")


# ---------------------------------------------------------------------------
# Published truth loader
# ---------------------------------------------------------------------------


def _parse_bool(value: str) -> bool:
    v = value.strip().lower()
    if v in ("true", "1", "yes", "y"):
        return True
    if v in ("false", "0", "no", "n"):
        return False
    raise ValueError(f"unparseable boolean: {value!r}")


def load_truth_table(csv_path: Path | str | None = None) -> list[PublishedTruth]:
    """Read ``data/published_traverse_data.csv`` (flown rovers only)."""
    path = Path(csv_path) if csv_path else DEFAULT_TRUTH_CSV
    rows: list[PublishedTruth] = []
    with path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(
                PublishedTruth(
                    rover_name=row["rover_name"],
                    scenario_name=row["scenario_name"],
                    traverse_m_published=float(row["traverse_m_published"]),
                    traverse_m_low=float(row["traverse_m_low"]),
                    traverse_m_high=float(row["traverse_m_high"]),
                    peak_solar_power_w_published=float(row["peak_solar_power_w_published"]),
                    peak_solar_power_w_low=float(row["peak_solar_power_w_low"]),
                    peak_solar_power_w_high=float(row["peak_solar_power_w_high"]),
                    thermal_survival_published=_parse_bool(row["thermal_survival_published"]),
                    mission_duration_published_days=float(row["mission_duration_published_days"]),
                    citation=row["citation"],
                    notes=row["notes"],
                )
            )
    return rows


def truth_by_rover(rover_name: str, csv_path: Path | str | None = None) -> PublishedTruth:
    """Fetch the published-truth row for one rover (must be flown)."""
    for row in load_truth_table(csv_path):
        if row.rover_name == rover_name:
            return row
    raise KeyError(
        f"no published-truth row for rover {rover_name!r}. "
        "(Truth rows are only stored for flown rovers; design-target "
        "rovers like MoonRanger/Rashid-1 are intentionally absent.)"
    )
