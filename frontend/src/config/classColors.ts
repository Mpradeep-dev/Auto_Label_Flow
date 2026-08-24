/**
 * Deterministic class index -> colour. Index 0 always gets the same hue no
 * matter what that class is named — the mapping is stable as a project's
 * class list changes (e.g. detect_v1's class 0 is "ball" today; a
 * differently-trained model's class 0 might be anything). Never keyed by
 * class NAME: names come from whatever model is loaded, this file must not
 * assume any of them.
 *
 * Red (`accent`) is deliberately absent from this palette — it is reserved
 * exclusively for the suspicious-flag border (see DESIGN.md "Box colour
 * language"). A flagged box gets a red outline ON TOP OF its class colour,
 * never instead of it.
 */
const PALETTE = [
  "#2563EB", // blue
  "#F59E0B", // amber
  "#10B981", // emerald
  "#8B5CF6", // violet
  "#06B6D4", // cyan
  "#EC4899", // pink
] as const;

export function classColor(classIndex: number): string {
  return PALETTE[classIndex % PALETTE.length];
}

/** 1-9 keyboard shortcuts map to a project's classes in list order. */
export function classShortcutKey(classIndex: number): string | null {
  return classIndex < 9 ? String(classIndex + 1) : null;
}
