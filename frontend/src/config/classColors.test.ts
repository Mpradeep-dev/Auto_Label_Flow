import { describe, expect, it } from "vitest";
import { classColor, classShortcutKey, classTextColor } from "./classColors";

function contrastRatio(hexA: string, hexB: string): number {
  function luminance(hex: string): number {
    const [r, g, b] = [hex.slice(1, 3), hex.slice(3, 5), hex.slice(5, 7)].map((h) => {
      const c = parseInt(h, 16) / 255;
      return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
    });
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
  }
  const [lA, lB] = [luminance(hexA), luminance(hexB)];
  return (Math.max(lA, lB) + 0.05) / (Math.min(lA, lB) + 0.05);
}

describe("classColor", () => {
  it("is deterministic for the same index", () => {
    expect(classColor(1)).toBe(classColor(1));
  });

  it("gives different colours to different indices within the palette", () => {
    expect(classColor(0)).not.toBe(classColor(1));
    expect(classColor(1)).not.toBe(classColor(2));
  });

  it("wraps around once the palette is exhausted, deterministically", () => {
    // Whatever the palette length N is, index N must equal index 0 —
    // this is the "stable regardless of class count" contract, not a
    // specific palette size.
    let paletteLength = 1;
    while (classColor(paletteLength) !== classColor(0)) paletteLength++;
    expect(classColor(paletteLength)).toBe(classColor(0));
  });

  it("never returns the reserved accent red", () => {
    for (let i = 0; i < 20; i++) {
      expect(classColor(i).toUpperCase()).not.toBe("#FF3000");
    }
  });
});

describe("classTextColor", () => {
  // Regression coverage for audit finding FE-19: text on a class swatch
  // used to always be black regardless of background, which computed to
  // ~4.1:1 on the default blue — below WCAG AA's 4.5:1 for small bold text.
  it("meets WCAG AA (4.5:1) contrast against its own swatch for every palette entry", () => {
    for (let i = 0; i < 12; i++) {
      const bg = classColor(i);
      const fg = classTextColor(i);
      expect(contrastRatio(bg, fg)).toBeGreaterThanOrEqual(4.5);
    }
  });

  it("is deterministic and only ever black or white", () => {
    for (let i = 0; i < 12; i++) {
      expect(["#000000", "#FFFFFF"]).toContain(classTextColor(i));
      expect(classTextColor(i)).toBe(classTextColor(i));
    }
  });
});

describe("classShortcutKey", () => {
  it("maps class index to a 1-based key for the first 9 classes", () => {
    expect(classShortcutKey(0)).toBe("1");
    expect(classShortcutKey(8)).toBe("9");
  });

  it("returns null beyond the 9th class", () => {
    expect(classShortcutKey(9)).toBeNull();
  });
});
