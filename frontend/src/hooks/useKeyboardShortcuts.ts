import { useEffect } from "react";

export interface ShortcutHandlers {
  add: () => void;
  delete: () => void;
  edit: () => void;
  prev: () => void;
  next: () => void;
  approve: () => void;
  save: () => void;
  zoom: () => void;
  fit: () => void;
  /** class1..class9 map to a project's classes in list order — only the
   * classes that exist get a working shortcut. */
  setClassByIndex: (index: number) => void;
}

/** Global handler for the declarative SHORTCUTS map (see hooks/shortcuts.ts).
 * Ignored while focus is inside a text/number input so typing coordinates
 * or a project name doesn't fight with single-letter shortcuts. */
export function useKeyboardShortcuts(handlers: ShortcutHandlers, enabled: boolean) {
  useEffect(() => {
    if (!enabled) return;

    function onKeyDown(e: KeyboardEvent) {
      const target = e.target as HTMLElement | null;
      if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable)) {
        return;
      }

      switch (e.key) {
        case "a":
        case "A":
          handlers.add();
          break;
        case "d":
        case "D":
          handlers.delete();
          break;
        case "e":
        case "E":
          handlers.edit();
          break;
        case "ArrowLeft":
          handlers.prev();
          break;
        case "ArrowRight":
          handlers.next();
          break;
        case " ":
          e.preventDefault();
          handlers.approve();
          break;
        case "s":
        case "S":
          e.preventDefault();
          handlers.save();
          break;
        case "z":
        case "Z":
          handlers.zoom();
          break;
        case "f":
        case "F":
          handlers.fit();
          break;
        default: {
          const n = Number(e.key);
          if (Number.isInteger(n) && n >= 1 && n <= 9) {
            handlers.setClassByIndex(n - 1);
          }
        }
      }
    }

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [handlers, enabled]);
}
