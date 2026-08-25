import { create } from "zustand";
import type { Annotation } from "@/types";

export type CanvasMode = "select" | "draw";

type PendingBox = Pick<Annotation, "x1" | "y1" | "x2" | "y2">;

interface AnnotationStoreState {
  annotations: Annotation[];
  selectedId: string | null;
  mode: CanvasMode;
  drawClassId: number;
  // A box whose geometry is drawn but not yet persisted — class choice now
  // happens AFTER drawing, not before, so the box sits here awaiting a
  // class pick (or a cancel) instead of being written straight to the API.
  pendingBox: PendingBox | null;

  setAnnotations: (annotations: Annotation[]) => void;
  selectAnnotation: (id: string | null) => void;
  upsertAnnotation: (annotation: Annotation) => void;
  removeAnnotation: (id: string) => void;
  setMode: (mode: CanvasMode) => void;
  setDrawClassId: (classId: number) => void;
  setPendingBox: (box: PendingBox | null) => void;
}

/**
 * Holds ONLY the current image's box data + selection/mode — ephemeral,
 * per-image interaction state. Server-derived data (image list, dataset
 * metadata) stays in TanStack Query, deliberately not merged into this
 * store (see DESIGN plan "State management").
 */
export const useAnnotationStore = create<AnnotationStoreState>((set) => ({
  annotations: [],
  selectedId: null,
  mode: "select",
  drawClassId: 0,
  pendingBox: null,

  // Switching images (this fires on every image navigation) must not carry
  // a half-drawn, unclassified box from the previous image along with it.
  setAnnotations: (annotations) => set({ annotations, pendingBox: null }),
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
  setMode: (mode) =>
    set((state) => ({ mode, selectedId: mode === "draw" ? null : state.selectedId })),
  setDrawClassId: (classId) => set({ drawClassId: classId }),
  setPendingBox: (box) => set({ pendingBox: box }),
}));
