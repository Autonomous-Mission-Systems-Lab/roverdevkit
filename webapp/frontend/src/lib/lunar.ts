/**
 * Lunar calendar constants for mission-duration UX.
 *
 * Mirrors `roverdevkit.power.solar.LUNAR_SYNODIC_DAY_HOURS` (29.530589 h
 * per synodic day). Canonical scenario YAMLs often use ~14 Earth days as
 * "one lunar day" meaning one *sunlit half-period* at non-polar latitudes
 * (Pragyan, Rashid), not a full synodic lunation.
 */

/** Full synodic lunar day in Earth days (~29.53 d). */
export const LUNAR_SYNODIC_DAYS = 29.530589;

/** Sunlit half-period at non-polar latitudes (~14.77 Earth days). */
export const SUNLIT_HALF_DAYS = LUNAR_SYNODIC_DAYS / 2.0;

/** Slider bounds for the mission-duration override (Earth days). */
export const MISSION_DURATION_BOUNDS = {
  min: 1,
  max: 60,
  step: 0.5,
} as const;

export function formatLunationContext(days: number): string {
  const lunations = days / LUNAR_SYNODIC_DAYS;
  const sunlitPeriods = days / SUNLIT_HALF_DAYS;
  if (lunations >= 0.95 && lunations <= 1.05) {
    return "≈ 1 full lunation";
  }
  if (sunlitPeriods >= 0.95 && sunlitPeriods <= 1.05) {
    return "≈ 1 sunlit half-period";
  }
  if (lunations >= 1.5) {
    return `≈ ${lunations.toFixed(1)} lunations`;
  }
  return `≈ ${sunlitPeriods.toFixed(1)} sunlit half-periods`;
}
