/**
 * Per-rover marker colors used by the registry-overlay picker and
 * the prediction chart. Hand-picked from a colorblind-safe
 * categorical palette; falls back to a hash if a rover is added
 * without an explicit entry.
 */

const ROVER_COLORS: Record<string, string> = {
  Pragyan: "#d97706",
  "Yutu-2": "#059669",
  MoonRanger: "#dc2626",
  "Rashid-1": "#7c3aed",
};

const FALLBACK_COLORS = ["#0ea5e9", "#84cc16", "#ec4899", "#f59e0b"];

export function roverColor(name: string): string {
  if (name in ROVER_COLORS) return ROVER_COLORS[name];
  let hash = 0;
  for (let i = 0; i < name.length; i += 1) {
    hash = (hash * 31 + name.charCodeAt(i)) >>> 0;
  }
  return FALLBACK_COLORS[hash % FALLBACK_COLORS.length];
}
