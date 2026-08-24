import { describe, expect, it } from "vitest";
import { classColor, classShortcutKey } from "./classColors";

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

describe("classShortcutKey", () => {
  it("maps class index to a 1-based key for the first 9 classes", () => {
    expect(classShortcutKey(0)).toBe("1");
    expect(classShortcutKey(8)).toBe("9");
  });

  it("returns null beyond the 9th class", () => {
    expect(classShortcutKey(9)).toBeNull();
  });
});
