import { useEffect } from "react";

export interface ShortcutHandlers {
  drawBbox: () => void;
  drawPolygon: () => void;
  drawSam: () => void;
  delete: () => void;
  undo: () => void;
  prev: () => void;
  next: () => void;
  approve: () => void;
  save: () => void;
  zoom: () => void;
  zoomOut: () => void;
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

      // Ctrl (Windows/Linux) or Cmd (Mac) + Z is undo — checked before the
      // plain-key switch below so it doesn't also trigger zoom-in, which
      // bare "z" is bound to.
      if ((e.ctrlKey || e.metaKey) && (e.key === "z" || e.key === "Z")) {
        e.preventDefault();
        handlers.undo();
        return;
      }

      switch (e.key) {
        case "b":
        case "B":
          handlers.drawBbox();
          break;
        case "p":
        case "P":
          handlers.drawPolygon();
          break;
        case "m":
        case "M":
          handlers.drawSam();
          break;
        case "d":
        case "D":
          handlers.delete();
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
          handlers.zoom();
          break;
        case "Z":
          // Shift+Z (Shift turns "z" into "Z") — plain Z above is zoom-in;
          // previously there was no keyboard zoom-out at all.
          handlers.zoomOut();
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
