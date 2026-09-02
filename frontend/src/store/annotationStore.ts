import { create } from "zustand";
import type { Annotation } from "@/types";

export type CanvasMode = "select" | "draw-bbox" | "draw-polygon" | "draw-sam";

export type PendingShape =
  | ({ shape_type: "BBOX" } & Pick<Annotation, "x1" | "y1" | "x2" | "y2">)
  | { shape_type: "POLYGON"; points: [number, number][] };

interface AnnotationStoreState {
  annotations: Annotation[];
  selectedId: string | null;
  mode: CanvasMode;
  drawClassId: number;
  // A shape whose geometry is drawn but not yet persisted — class choice now
  // happens AFTER drawing, not before, so the shape sits here awaiting a
  // class pick (or a cancel) instead of being written straight to the API.
  pendingShape: PendingShape | null;

  setAnnotations: (annotations: Annotation[]) => void;
  selectAnnotation: (id: string | null) => void;
  upsertAnnotation: (annotation: Annotation) => void;
  removeAnnotation: (id: string) => void;
  setMode: (mode: CanvasMode) => void;
  setDrawClassId: (classId: number) => void;
  setPendingShape: (shape: PendingShape | null) => void;
}

/**
 * Holds ONLY the current image's annotation data + selection/mode —
 * ephemeral, per-image interaction state. Server-derived data (image list,
 * dataset metadata) stays in TanStack Query, deliberately not merged into
 * this store (see DESIGN plan "State management").
 */
export const useAnnotationStore = create<AnnotationStoreState>((set) => ({
  annotations: [],
  selectedId: null,
  mode: "select",
  drawClassId: 0,
  pendingShape: null,

  // Switching images (this fires on every image navigation) must not carry
  // a half-drawn, unclassified shape from the previous image along with it.
  setAnnotations: (annotations) => set({ annotations, pendingShape: null }),
  selectAnnotation: (id) => set({ selectedId: id }),
  upsertAnnotation: (annotation) =>
    set((state) => {
      const exists = state.annotations.some((a) => a.id === annotation.id);
      return {
        annotations: exists
          ? state.annotations.map((a) => (a.id === annotation.id ? annotation : a))
          : [...state.annotations, annotation],
      };
    }),
  removeAnnotation: (id) =>
    set((state) => ({
      annotations: state.annotations.filter((a) => a.id !== id),
      selectedId: state.selectedId === id ? null : state.selectedId,
    })),
  // Switching tools (or returning to select) always drops any half-drawn
  // shape from whichever tool was active before — a polygon mid-ring
  // shouldn't survive a switch to the box tool. Callers that just finished
  // a shape call setMode("select") *before* handing the finished geometry
  // up (see AnnotationCanvas's finishDraw/finishPolygon), so this doesn't
  // clobber a shape that's about to become pending.
  setMode: (mode) =>
    set((state) => ({
      mode,
      selectedId: mode === "select" ? state.selectedId : null,
      pendingShape: null,
    })),
  setDrawClassId: (classId) => set({ drawClassId: classId }),
  setPendingShape: (shape) => set({ pendingShape: shape }),
}));
