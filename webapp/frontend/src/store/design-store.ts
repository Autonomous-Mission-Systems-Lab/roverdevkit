import { create } from "zustand";

import type { DesignVector, ScenarioName } from "@/types/api";

/**
 * Local UI state for the single-design panel.
 *
 * Pulled into Zustand rather than React component state so future
 * panels (sweep, Pareto explorer) can share the same "currently
 * selected design + scenario" without prop-drilling. TanStack Query
 * still owns *server* state (scenarios list, predict response, etc.).
 */

/**
 * Default design vector used as the form's starting point.
 *
 * Rough Yutu-2 / mid-class lunar micro-rover (matches the backend's
 * predict-test fixture). All eleven fields (schema v7) sit
 * comfortably inside the LHS bounds so the surrogate sees an
 * in-distribution input on first render.
 */
export const DEFAULT_DESIGN: DesignVector = {
  wheel_radius_m: 0.1,
  wheel_width_m: 0.1,
  grouser_height_m: 0.012,
  grouser_count: 14,
  n_wheels: 6,
  chassis_mass_kg: 20,
  wheelbase_m: 0.6,
  solar_area_m2: 0.5,
  battery_capacity_wh: 100,
  avionics_power_w: 15,
  peak_wheel_torque_nm: 1.5,
};

interface DesignState {
  design: DesignVector;
  scenarioName: ScenarioName;
  /**
   * Optional per-query override for `MissionScenario.operational_duty_cycle`.
   *
   * `null` means "use the scenario's calibrated default"; setting an
   * explicit number passes that value through to both `/evaluate` and
   * `/predict`. SCHEMA_VERSION v7_1 (v7_1 schema follow-on): δ_ops is
   * an LHS feature so the surrogate keeps calibrated PIs across the
   * entire slider range; pre-v7_1 any override forced an evaluator-
   * only fallback that suppressed the PI band. Reset to `null`
   * automatically when the user picks a different scenario so we
   * don't carry a polar-conservative δ_ops onto an equatorial-mare
   * traverse without realising it. Schema v7: this is the *only*
   * duty-cycle knob — `designed_duty_cycle` was removed from the
   * design vector after it turned out to do no engineering work in
   * the v6 mass model.
   */
  opsDutyOverride: number | null;
  /**
   * Optional per-query overrides for the schema-v9 payload mission
   * requirements (`MissionScenario.payload_mass_kg` / `payload_power_w`).
   *
   * `null` means "use the scenario's class-typical default"; an explicit
   * number is passed through to `/evaluate`, `/predict`, `/sweep`, and
   * `/optimize`. Reset to `null` automatically when the user switches
   * scenario so a heavy-payload override doesn't silently carry onto a
   * different mission class. Both are LHS-sampled surrogate inputs over
   * [0, 30] so the surrogate keeps calibrated PIs across the whole range.
   */
  payloadMassOverride: number | null;
  payloadPowerOverride: number | null;
  /**
   * Optional per-query override for `MissionScenario.mission_duration_earth_days`.
   *
   * Sets the simulation window (solar averaging, energy budget, thermal
   * exposure). Reset to `null` on scenario change so a polar 30 d window
   * doesn't carry onto a mare traverse without realising it.
   */
  missionDurationOverride: number | null;
  /** Names of registry rovers whose predictions should be overlaid on the chart. */
  overlayRovers: string[];
  setDesignField: <K extends keyof DesignVector>(
    key: K,
    value: DesignVector[K],
  ) => void;
  setDesign: (design: DesignVector) => void;
  setScenario: (name: ScenarioName) => void;
  setOpsDutyOverride: (value: number | null) => void;
  clearOpsDutyOverride: () => void;
  setPayloadMassOverride: (value: number | null) => void;
  setPayloadPowerOverride: (value: number | null) => void;
  clearPayloadOverrides: () => void;
  setMissionDurationOverride: (value: number | null) => void;
  clearMissionDurationOverride: () => void;
  resetDesign: () => void;
  toggleOverlayRover: (name: string) => void;
  clearOverlayRovers: () => void;
}

export const useDesignStore = create<DesignState>()((set) => ({
  design: DEFAULT_DESIGN,
  scenarioName: "equatorial_mare_traverse",
  opsDutyOverride: null,
  payloadMassOverride: null,
  payloadPowerOverride: null,
  missionDurationOverride: null,
  overlayRovers: [],
  setDesignField: (key, value) =>
    set((state) => ({ design: { ...state.design, [key]: value } })),
  setDesign: (design) => set({ design }),
  setScenario: (name) =>
    set({
      scenarioName: name,
      opsDutyOverride: null,
      payloadMassOverride: null,
      payloadPowerOverride: null,
      missionDurationOverride: null,
    }),
  setOpsDutyOverride: (value) => set({ opsDutyOverride: value }),
  clearOpsDutyOverride: () => set({ opsDutyOverride: null }),
  setPayloadMassOverride: (value) => set({ payloadMassOverride: value }),
  setPayloadPowerOverride: (value) => set({ payloadPowerOverride: value }),
  clearPayloadOverrides: () =>
    set({ payloadMassOverride: null, payloadPowerOverride: null }),
  setMissionDurationOverride: (value) => set({ missionDurationOverride: value }),
  clearMissionDurationOverride: () => set({ missionDurationOverride: null }),
  resetDesign: () =>
    set({
      design: DEFAULT_DESIGN,
      opsDutyOverride: null,
      payloadMassOverride: null,
      payloadPowerOverride: null,
      missionDurationOverride: null,
    }),
  toggleOverlayRover: (name) =>
    set((state) => ({
      overlayRovers: state.overlayRovers.includes(name)
        ? state.overlayRovers.filter((r) => r !== name)
        : [...state.overlayRovers, name],
    })),
  clearOverlayRovers: () => set({ overlayRovers: [] }),
}));
