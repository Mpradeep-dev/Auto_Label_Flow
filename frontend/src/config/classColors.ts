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

function srgbChannelToLinear(c8bit: number): number {
  const c = c8bit / 255;
  return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}

/** WCAG relative luminance of a #rrggbb color. */
function relativeLuminance(hex: string): number {
  const r = srgbChannelToLinear(parseInt(hex.slice(1, 3), 16));
  const g = srgbChannelToLinear(parseInt(hex.slice(3, 5), 16));
  const b = srgbChannelToLinear(parseInt(hex.slice(5, 7), 16));
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

/** WCAG contrast ratio between two #rrggbb colors — always >= 1. */
function contrastRatio(hexA: string, hexB: string): number {
  const lA = relativeLuminance(hexA);
  const lB = relativeLuminance(hexB);
  const lighter = Math.max(lA, lB);
  const darker = Math.min(lA, lB);
  return (lighter + 0.05) / (darker + 0.05);
}

/** Text color to put ON TOP of `classColor(classIndex)` — whichever of
 * black/white gives the higher WCAG contrast ratio against that specific
 * swatch. Audit finding FE-19: the canvas label always used black text
 * regardless of background; on the default blue swatch (#2563EB) that's
 * ~4.1:1 — below WCAG AA's 4.5:1 for the small bold label text used, and
 * the class every new project starts with. A simple luminance-midpoint
 * heuristic isn't accurate enough across this exact palette (it picks
 * white for cyan, whose real contrast ratios are 2.4:1 white vs. 8.7:1
 * black), so this compares actual contrast ratios instead of estimating. */
export function classTextColor(classIndex: number): "#000000" | "#FFFFFF" {
  const bg = classColor(classIndex);
  const blackContrast = contrastRatio(bg, "#000000");
  const whiteContrast = contrastRatio(bg, "#FFFFFF");
  return blackContrast >= whiteContrast ? "#000000" : "#FFFFFF";
}

/** 1-9 keyboard shortcuts map to a project's classes in list order. */
export function classShortcutKey(classIndex: number): string | null {
  return classIndex < 9 ? String(classIndex + 1) : null;
}
