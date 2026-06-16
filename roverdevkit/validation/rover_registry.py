"""Published-rover design vectors and mission scenarios.

This module codifies the lunar rovers we compare the evaluator and the
surrogate against as
:class:`DesignVector` + :class:`MissionScenario` pairs, plus the
published truth numbers in ``data/published_traverse_data.csv``.

Two-tier registry
-----------------
The registry is split into two tiers via :attr:`RoverRegistryEntry.is_flown`:

- **Flown** (``is_flown=True``): rovers with actual ground-truth flight
  data. Used by:

  * Layer-0 truth comparison (real-rover validation gate,
    :func:`roverdevkit.validation.rover_comparison.compare_all`),
    which scores the evaluator vs published traverse / peak-solar /
    thermal data.
  * Layer-1 surrogate sanity check (baseline-surrogate,
    :func:`roverdevkit.surrogate.baselines.predict_for_registry_rovers`).

  Currently: **Pragyan** (Chandrayaan-3, 2023), **Yutu-2** (Chang'e-4,
  2019).

- **Design-target** (``is_flown=False``): well-spec'd lunar micro-rover
  designs that did not fly (lander loss or still in development). Used
  only for Layer-1 surrogate sanity. Layer-0 truth comparison is
  skipped because there's no ground-truth flight data.

  Currently: **MoonRanger** (CMU/Astrobotic, in development),
  **Rashid-1** (MBRSC/UAE, lost on Hakuto-R Mission 1, 2023),
  **Tenacious** (iSpace/HAKUTO-R Mission 2, lost on landing failure,
  June 2025), **CADRE-unit** (NASA JPL, ultra-micro flotilla
  technology demonstration, 2024-2025 launch / deployment window;
  treated as design-target until a published surface-mission report
  is available).

Pending (not yet in the registry)
---------------------------------
- **MAPP** (Lunar Outpost, deployed on IM-2 / Athena Feb 2025) —
  potential flown entry pending consolidation of published specs
  (mass / wheels / solar). Not added in the current pass to avoid
  shipping imputed numbers without a primary citation.

Class scope (2026-05-27 widening)
---------------------------------
The schema's ``chassis_mass_kg`` / ``peak_wheel_torque_nm`` /
``battery_capacity_wh`` floors were lowered (to 0.5 kg, 0.05 Nm,
5 Wh respectively) so CADRE (~0.8 kg chassis) and Tenacious (~2 kg
chassis) sit inside the schema's valid design space rather than at
or below the previous floor. The v4 LHS surrogate's training support
remains the older ``(3.0, 0.3, 20.0)`` floors; the new ultra-micro
entries are therefore OOD for the Layer-1 surrogate sanity check
until the v5 LHS regeneration. The rediscovery harness runs against
the corrected evaluator (not the surrogate) and is unaffected.

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
  :func:`list_scenarios` so webapp sweeps never pick them up.

Every design-vector field that is not directly published has an entry
in :attr:`RoverRegistryEntry.imputation_notes`; these notes mirror the
pattern from the mass-validation mass validation set for consistency.

``chassis_mass_kg`` semantics (registry-wide audit, 2026-05-27)
---------------------------------------------------------------
The schema's :attr:`roverdevkit.schema.DesignVector.chassis_mass_kg`
field represents the **structural-chassis-only mass**, not the
full-up rover mass. The bottom-up parametric mass model in
:mod:`roverdevkit.mass.parametric_mers` adds wheels, drive motors,
solar panels, battery pack, avionics, harness, thermal control, and
a 25 % AIAA S-120A growth margin on top of the chassis input. The
schema-correct rule of thumb (matching ``data/mass_validation_set.csv``)
is **chassis ≈ 35-40 % of full-up rover mass**.

Audit of the 6-rover registry:

==============  ==================  ==================  =======
Rover           registry chassis    published total     chassis %
==============  ==================  ==================  =======
Pragyan         10.0 kg             26 kg               38 %
Yutu-2          35.0 kg             135 kg              26 % (chassis ex-payload; payload absorbed elsewhere)
Tenacious       2.0 kg              5 kg                40 %
CADRE-unit      0.8 kg              2 kg                40 %
MoonRanger      4.5 kg              13 kg               35 %  (back-solved 2026-05-27; was 13.0 — buggy)
Rashid-1        3.5 kg              10 kg               35 %  (back-solved 2026-05-27; was 10.0 — buggy)
==============  ==================  ==================  =======

The MoonRanger and Rashid-1 chassis values were incorrectly set to
their published full-up totals before the 2026-05-27 audit. That
inflated the bottom-up sum by ~2× for those two rovers and biased
their Layer-4 / Layer-5 validation outputs (Layer-4 predicted-vs-published
mass error, Layer-5 mass-budget constraint). Back-solved values
now land within the 35-40 % class band.

Payload as a mission requirement (schema v9)
--------------------------------------------
Scientific payload is no longer folded into ``chassis_mass_kg``. Each
rover's published instrument-suite mass is carried on its
per-rover validation scenario YAML
(``payload_mass_kg`` / ``payload_power_w`` on
:class:`roverdevkit.schema.MissionScenario`) and added to the total by
the evaluator as a line item outside the dry-mass growth margin. The
``chassis_mass_kg`` values in this registry stay structural-only and
unchanged; adding the scenario payload makes each rover's *evaluated*
total mass land closer to its published full-up mass (e.g. Pragyan
bottom-up bus ~22 kg + 3.5 kg payload ≈ 26 kg published; Yutu-2
+25 kg payload). ``payload_power_w`` is held at 0 on the per-rover
validation scenarios because the published traverse / peak-solar /
thermal truth was measured during mobility windows with the science
instruments powered down — so the Layer-0 mobility-validation gate
sees payload mass (always carried) but not instrument power draw. The
rediscovery harness (Layer-5) instead forwards each target rover's
payload as a per-call override to both the rover re-evaluation and
every NSGA-II candidate so the dominance comparison is
apples-to-apples.
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
    panel_tilt_deg
        Solar-array tilt off the chassis horizontal plane, deg in
        [0, 90]. ``0`` is a flat top-mounted panel (collector facing
        zenith), the most common geometry for mid-latitude / mare
        rovers whose noon sun reaches 30-60 deg elevation. Polar
        rovers carry deployable masts or articulated panels that
        face the low-elevation polar sun; for those, ``panel_tilt_deg``
        is set close to ``90 - lat`` so the surface normal points
        toward the noon sun. Used jointly with ``panel_azimuth_deg``;
        ignored when ``panel_tilt_deg == 0``.

        Validation-gate calibration note: the flown rovers
        (Pragyan, Yutu-2) are kept at ``panel_tilt_deg = 0`` because
        their published peak-solar-power truth bands in
        ``data/published_traverse_data.csv`` were measured as
        operational-average values (averaged over the rover's
        actual driving orientation, not at a sun-tracking attitude
        hold). Their canonical-scenario validation gates assume
        horizontal-equivalent panel pointing.
    panel_azimuth_deg
        Direction the tilted panel faces (clockwise from north),
        deg in [0, 360). For rovers in the southern hemisphere
        (negative latitude) the noon sun is in the north so panels
        face ``0`` deg; for northern hemisphere rovers the noon sun
        is in the south so panels face ``180`` deg. Default 180
        matches the SMAD / Patel southern-array convention
        (mid-northern latitude rover). Ignored when
        ``panel_tilt_deg == 0`` because cosine-of-incidence then
        depends only on elevation.
    is_flown
        Whether the rover successfully deployed and produced
        ground-truth flight data. Drives whether the entry participates
        in the Layer-0 truth comparison (see module docstring).
    mass_model_in_regime
        ``True`` (default) when the rover sits inside the bottom-up
        mass model's specific-mass calibration regime (5-50 kg total
        mass; see :class:`MassModelParams` and
        :mod:`roverdevkit.mass.validation`). ``False`` for ultra-micro
        rovers below the calibration floor (currently CADRE-unit at
        ~2 kg total) where the model's fixed-cost terms over-predict
        total mass by ~100 %. Out-of-regime rovers participate in
        rediscovery and in the schema-bounds widening (A2) but are
        skipped by mass-dependent Layer-1 gates (stall-edge sanity
        in particular) because the over-predicted modelled mass
        artificially pushes torque demand past the rover's actual
        capability.
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
    mass_model_in_regime: bool = True
    panel_tilt_deg: float = 0.0
    panel_azimuth_deg: float = 180.0


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
    # Schema v6 (v6 schema update): nominal_speed_mps no longer a design
    # input (cruise speed is derived); drive_duty_cycle renamed to
    # designed_duty_cycle. Schema v7 (v7 schema follow-up):
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
    # - total rover mass: 13 kg (full-up flight mass, all subsystems)
    # - n_wheels: 4
    # - max mechanical speed: 0.07 m/s ("7 cm/sec")
    # - mission duration: 8 Earth days
    # - rover length: ~0.65 m (half-length 0.325 m used for FOV calc)
    # - camera height: 0.25 m
    # - lunar South Pole, no RHU (operates in single daylight period).
    #
    # Imputations (back-solve + class match to Rashid-1):
    # - chassis_mass_kg = 4.5: back-solved so the bottom-up
    #   parametric mass model (chassis + wheels + motors + solar +
    #   battery + avionics + harness + thermal + 25 % margin)
    #   yields a total close to the published 13 kg full-up mass.
    #   chassis ≈ 35 % of total, consistent with the
    #   ``data/mass_validation_set.csv`` convention and with the
    #   other micro-rover registry entries (Pragyan 38 %, Tenacious
    #   40 %, CADRE 40 %). Note: the schema's ``chassis_mass_kg``
    #   field is the structural-chassis-only mass, NOT the published
    #   total — see the registry-wide audit note in the
    #   :func:`registry` docstring.
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
        chassis_mass_kg=4.5,  # back-solved from published 13 kg total
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
        # Mast-deployable solar array oriented toward the low polar
        # sun (Kumar et al. i-SAIRAS 2020 #5068; CMU MoonRanger labs
        # page imagery): the deployable mast tilts the panel so its
        # surface normal can track the sun across the horizon as the
        # rover drives. At lat=-85.0 the noon sun elevation is ~5
        # deg, so a panel tilt of 80 deg from horizontal keeps the
        # incidence angle near 0 deg - within the ~10 deg pointing
        # tolerance of a fixed-tilt approximation. Azimuth=0 because
        # MoonRanger's south-polar landing site has the noon sun in
        # the local north sky; rovers in the southern hemisphere
        # face their tilted panels toward azimuth=0.
        panel_tilt_deg=80.0,
        panel_azimuth_deg=0.0,
        imputation_notes=(
            "Cited: total mass (13 kg full-up), n_wheels (4), max mech "
            "speed (0.07 m/s), mission duration (8 d), no RHU. Imputed: "
            "chassis_mass_kg = 4.5 (back-solved so the bottom-up "
            "subsystem total matches the published 13 kg; "
            "schema-correct value is the structural chassis only); "
            "wheel radius/width and grousers (class-match to Rashid-1); "
            "wheelbase from published rover length; solar / battery / "
            "avionics from a power budget back-solve against the "
            "kilometer-per-day exploration target. panel_tilt_deg=80, "
            "panel_azimuth_deg=0: mast-deployable polar array oriented "
            "toward the low (~5 deg) noon sun at the south-polar "
            "landing site (Kumar et al. 2020). Without the tilt the "
            "horizontal-panel assumption would under-predict insolation "
            "by ~18x at lat=-85 and the rover would stall on its own "
            "scenario."
        ),
    )


def _rashid1_entry() -> RoverRegistryEntry:
    # Direct cites (Hurrell et al. 2025 SSR 221:37 wheel paper,
    # Els et al. LPSC 2021 #1905 instrumentation paper, ESA + Wikipedia
    # ELM page):
    # - total rover mass: 10 kg (full-up flight mass, all subsystems)
    # - n_wheels: 4
    # - wheel_radius_m: 0.10 ("radius of 100 mm")
    # - wheel_width_m: 0.08 ("width of 80 mm")
    # - grouser_height_m: 0.015 (15 mm flight grouser; Hurrell 2025
    #   distinguishes from the 20 mm closed-side test wheel)
    # - grouser_count: 14
    # - wheelbase_m: 0.50 (footprint 0.535 x 0.539 m per LPSC 2021)
    # - landing site: Atlas crater, Mare Frigoris (~47.5 N, 44.4 E)
    # - mission duration: 1 lunar day (~14 Earth days), no RHU.
    # - Hurrell 2025 used 0.02 m/s as the experimental drive velocity;
    #   that is now scenario-side context rather than a design input
    #   under the v6 schema (cruise speed is derived).
    #
    # Imputations:
    # - chassis_mass_kg = 3.5: back-solved so the bottom-up
    #   parametric mass model totals close to the published 10 kg
    #   full-up mass. chassis ≈ 35 % of total, matching the
    #   ``data/mass_validation_set.csv`` Rashid entry and the
    #   class-typical fraction across the rest of the registry
    #   (Pragyan 38 %, Tenacious 40 %, CADRE 40 %). The schema's
    #   ``chassis_mass_kg`` field is the structural-chassis-only
    #   mass; populating it with the published total would inflate
    #   the bottom-up sum to ~20 kg (the pre-fix bug). See the
    #   registry-wide audit note in the :func:`registry` docstring.
    # - solar_area_m2 = 0.25: 0.5 x 0.5 m chassis with deployable mast;
    #   flat array bound ~0.25 m^2. Power back-solve: at lunar noon
    #   ~47.5 N, 0.20 eff x 0.25 m^2 x 0.85 dust ~32 W peak, sufficient
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
        chassis_mass_kg=3.5,  # back-solved from published 10 kg total
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
            "total mass (10 kg full-up), n_wheels, wheel radius/width, "
            "grouser height (flight 15 mm) and count (14), wheelbase. "
            "Imputed: chassis_mass_kg = 3.5 (back-solved so the "
            "bottom-up subsystem total matches the published 10 kg; "
            "schema-correct value is the structural chassis only, ~35 % "
            "of total per class ROT); solar / battery / avionics from "
            "a power-budget back-solve against the science-payload "
            "inventory and single-lunar-day mission target. v6: "
            "peak_wheel_torque_nm=0.5 from v5-implicit hub-torque "
            "anchor at the published 10 kg / 4-wheel / R=0.10 m "
            "design point."
        ),
    )


def _tenacious_entry() -> RoverRegistryEntry:
    # Direct cites (iSpace HAKUTO-R Mission 2 mission overview and
    # press kit; iSpace mission docs; June 2025 lander-loss reporting):
    # - chassis_mass_kg total: 5 kg
    # - n_wheels: 4
    # - mission target: Mare Frigoris area, mid-northern latitude
    # - mission duration: ~1 lunar day (~14 Earth days), no RHU
    # - landing site lat ~60.5 N
    # - status: lander failed on descent (June 2025); rover never
    #   operated on the lunar surface (parallel to Rashid-1 on
    #   Hakuto-R M1)
    #
    # Imputations (back-solve + class match to Rashid-1):
    # - chassis_mass_kg = 2.0 (~40 % of total per ultra-micro class
    #   ROT; reflects published 5 kg total mass minus subsystem mass)
    # - wheel_radius_m = 0.06, wheel_width_m = 0.04: class-match to
    #   ultra-micro flight wheels visible in iSpace press imagery,
    #   smaller than Rashid-1 (0.10 / 0.08) by mass ratio
    # - grouser_height_m = 0.005, grouser_count = 12: small grousers
    #   visible in iSpace photos; class-typical 12-tooth pattern
    # - wheelbase_m = 0.30: small chassis ~0.4 m long minus wheel
    #   diameter ~0.12 m
    # - solar_area_m2 = 0.15: small body-mounted array on a
    #   ~0.4 x 0.4 m chassis
    # - battery_capacity_wh = 25: class-typical for a 5 kg lunar
    #   day-1 science demonstration rover
    # - avionics_power_w = 8: small flight computer + comms +
    #   science payload sensor suite during active ops
    # - peak_wheel_torque_nm = 0.10: v5-implicit anchor at 5 kg /
    #   4-wheel / R=0.06 / lunar gravity; just above the schema
    #   floor for ultra-micro rovers
    design = DesignVector(
        wheel_radius_m=0.06,
        wheel_width_m=0.04,
        grouser_height_m=0.005,
        grouser_count=12,
        n_wheels=4,
        chassis_mass_kg=2.0,
        wheelbase_m=0.30,
        solar_area_m2=0.15,
        battery_capacity_wh=25.0,
        avionics_power_w=8.0,
        peak_wheel_torque_nm=0.10,
    )
    # Thermal: Tenacious carries no RHU and was designed for a single
    # lunar day at mid-northern latitude. MLI + heaters used in flight
    # but not modelled explicitly. Surface area scales with the small
    # chassis; balanced absorptivity for mid-latitude diurnal swing.
    thermal = ThermalArchitecture(
        surface_area_m2=0.12,
        absorptivity=0.30,
        rhu_power_w=0.0,
        hibernation_power_w=1.5,
    )
    return RoverRegistryEntry(
        rover_name="Tenacious",
        design=design,
        scenario=load_scenario("ispace_m2_tenacious"),
        gravity_m_per_s2=GRAVITY_MOON_M_PER_S2,
        thermal_architecture=thermal,
        panel_efficiency=0.28,  # modern triple-junction GaAs BOL
        panel_dust_factor=0.95,  # brand new array; short mission
        is_flown=False,
        imputation_notes=(
            "Cited (iSpace HAKUTO-R M2 mission overview, June 2025 "
            "lander-loss reporting): total mass (5 kg), n_wheels (4), "
            "mission target (Mare Frigoris, ~60.5 N), "
            "no RHU, 1 lunar day target. Imputed: wheel radius / "
            "width / grousers (class-match to Rashid-1 scaled by "
            "mass ratio); wheelbase from small-chassis class typical; "
            "solar / battery / avionics / peak torque from a "
            "power-budget back-solve against the short-mission "
            "science-demonstration target. chassis_mass_kg=2.0 sits "
            "below the v4 LHS floor (3 kg) so Tenacious is OOD for "
            "the existing surrogate; covered by the 2026-05-27 "
            "schema floor widening and the planned v5 regen."
        ),
    )


def _cadre_unit_entry() -> RoverRegistryEntry:
    # Direct cites (Rothenbuchner et al. 2023 IEEE Aerospace #2300
    # "Cooperative Autonomous Distributed Robotic Exploration (CADRE)";
    # NASA/JPL CADRE project page; CADRE flotilla press materials):
    # - per-unit total mass: ~2 kg
    # - n_wheels: 4
    # - wheel_radius_m: 0.08 (Rothenbuchner 2023 / NASA-JPL CADRE press materials)
    # - wheelbase_m: 0.30
    # - solar_area_m2: 0.10 (small body-mounted array)
    # - mission target: lunar south pole region; multi-rover
    #   coordination demonstration with Nokia/Bell Labs LTE comms
    # - launch / deployment window: 2024-2025 onto a Commercial
    #   Lunar Payload Services lander
    # - status: as of registry snapshot, treated as `is_flown=False`
    #   design-target pending publication of the surface-mission
    #   report
    #
    # Imputations:
    # - chassis_mass_kg = 0.8 (~40 % of total; matches the
    #   mass_validation_set.csv entry for a CADRE flotilla unit)
    # - wheel_width_m = 0.04 (proportional to small wheel radius;
    #   roughly half the radius, class-typical for ultra-micro
    #   wire-spoke wheels)
    # - grouser_height_m = 0.0, grouser_count = 0: CADRE flotilla
    #   wheels are smooth wire-spoke rims (per JPL flotilla
    #   photography), not grousered. Conservative.
    # - battery_capacity_wh = 10: class-typical for a 2 kg
    #   flotilla member doing short coordinated drives
    # - avionics_power_w = 5: schema floor; CADRE units carry
    #   minimum flight-computer + LTE-comms power. Each unit
    #   has fewer sensors than a science-payload rover.
    # - peak_wheel_torque_nm = 0.06: v5-implicit anchor at
    #   2 kg / 4-wheel / R=0.08 / lunar gravity; near the
    #   schema floor (0.05 Nm) for ultra-micro flotilla rovers
    design = DesignVector(
        wheel_radius_m=0.08,
        wheel_width_m=0.04,
        grouser_height_m=0.0,
        grouser_count=0,
        n_wheels=4,
        chassis_mass_kg=0.8,
        wheelbase_m=0.30,
        solar_area_m2=0.10,
        battery_capacity_wh=10.0,
        avionics_power_w=5.0,
        peak_wheel_torque_nm=0.06,
    )
    # Thermal: CADRE units carry no RHU and operate only during
    # lunar daylight at the south pole landing site. Small chassis,
    # low absorptivity to keep hot-case rejection manageable in
    # near-continuous low-elevation sun.
    thermal = ThermalArchitecture(
        surface_area_m2=0.06,
        absorptivity=0.20,
        rhu_power_w=0.0,
        hibernation_power_w=0.5,
    )
    return RoverRegistryEntry(
        rover_name="CADRE-unit",
        design=design,
        scenario=load_scenario("cadre_polar_unit"),
        gravity_m_per_s2=GRAVITY_MOON_M_PER_S2,
        thermal_architecture=thermal,
        panel_efficiency=0.28,  # modern triple-junction GaAs BOL
        panel_dust_factor=0.95,  # brand new array; short mission
        is_flown=False,
        # Articulated top-deck panel that erects toward the low
        # polar sun during stationary "sun-baking" charging cycles
        # (Rothenbuchner et al. 2023; JPL CADRE flotilla imagery).
        # Same fixed-tilt approximation as MoonRanger: tilt=80 deg
        # at the south-polar landing site (lat=-85, noon elevation
        # ~5 deg). Azimuth=0 because the noon sun is in the local
        # north sky for southern-hemisphere rovers. Without the
        # tilt the horizontal-panel assumption under-predicts
        # insolation by ~18x at lat=-85 and even the optimiser
        # cannot find a feasible CADRE-class polar design within
        # the 2-kg mass budget.
        panel_tilt_deg=80.0,
        panel_azimuth_deg=0.0,
        imputation_notes=(
            "Cited (Rothenbuchner et al. 2023 IEEE Aerospace #2300; "
            "NASA/JPL CADRE project page; CADRE press materials): "
            "per-unit total mass (2 kg), n_wheels (4), wheel radius "
            "(0.08 m), wheelbase (0.30 m), solar area (0.10 m^2), "
            "south polar mission target, no RHU. Imputed: chassis "
            "mass (0.8 kg, ~40 % of total per ultra-micro class "
            "ROT); wheel width from class-typical aspect ratio; "
            "grousers absent (smooth wire-spoke rims in JPL "
            "imagery); battery and avionics from a power-budget "
            "back-solve against short coordinated-demonstration "
            "drives; peak torque from the v5-implicit hub-torque "
            "anchor. Six of the eleven design fields sit below the "
            "v4 LHS bounds (chassis 0.8 < 3, battery 10 < 20, "
            "avionics at floor, torque 0.06 < 0.3, plus solar at "
            "floor, no grousers); the entry is OOD for the v4 "
            "surrogate, covered by the 2026-05-27 schema floor "
            "widening and the planned v5 regen. The bottom-up mass "
            "model's specific-mass constants are calibrated for the "
            "5-50 kg micro-rover class and over-predict CADRE's "
            "2 kg total by ~100 % (fixed-cost terms dominate at "
            "sub-5 kg); CADRE is therefore reported as out-of-regime "
            "for the mass-model gate (in_class=False in "
            "data/mass_validation_set.csv) even though it is in the "
            "design-space class."
        ),
        mass_model_in_regime=False,
    )


# ---------------------------------------------------------------------------
# Registry accessors
# ---------------------------------------------------------------------------


def registry() -> tuple[RoverRegistryEntry, ...]:
    """Return the frozen tuple of all registry entries (flown + design-target).

    Use this for Layer-1 surrogate sanity checks (baseline-surrogate+). For Layer-0
    truth comparisons, use :func:`flown_registry` instead.
    """
    return (
        _pragyan_entry(),
        _yutu2_entry(),
        _moonranger_entry(),
        _rashid1_entry(),
        _tenacious_entry(),
        _cadre_unit_entry(),
    )


def flown_registry() -> tuple[RoverRegistryEntry, ...]:
    """Return only the rovers that successfully deployed and flew.

    Used by the real-rover validation gate
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
