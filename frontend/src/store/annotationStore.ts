import { create } from "zustand";
import type { Annotation } from "@/types";

export type CanvasMode = "select" | "draw";

interface AnnotationStoreState {
  annotations: Annotation[];
  selectedId: string | null;
  mode: CanvasMode;
  drawClassId: number;

  setAnnotations: (annotations: Annotation[]) => void;
  selectAnnotation: (id: string | null) => void;
  upsertAnnotation: (annotation: Annotation) => void;
  removeAnnotation: (id: string) => void;
  setMode: (mode: CanvasMode) => void;
  setDrawClassId: (classId: number) => void;
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

  setAnnotations: (annotations) => set({ annotations }),
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
}));
