import { beforeEach, describe, expect, it } from "vitest";
import { useAnnotationStore } from "./annotationStore";
import type { Annotation } from "@/types";

function makeAnnotation(overrides: Partial<Annotation> = {}): Annotation {
  return {
    id: "a1",
    image_id: "img1",
    class_id: 1,
    class_name: "cone",
    x1: 0.1,
    y1: 0.1,
    x2: 0.2,
    y2: 0.2,
    confidence: 0.7,
    source: "AUTO",
    review_status: "PENDING",
    revision_seq: 1,
    created_at: "",
    updated_at: "",
    ...overrides,
  };
}

beforeEach(() => {
  useAnnotationStore.setState({ annotations: [], selectedId: null, mode: "select", drawClassId: 0 });
});

describe("annotationStore", () => {
  it("setAnnotations replaces the whole list", () => {
    useAnnotationStore.getState().setAnnotations([makeAnnotation()]);
    expect(useAnnotationStore.getState().annotations).toHaveLength(1);
  });

  it("upsertAnnotation adds a new annotation", () => {
    useAnnotationStore.getState().upsertAnnotation(makeAnnotation({ id: "a2" }));
    expect(useAnnotationStore.getState().annotations.map((a) => a.id)).toEqual(["a2"]);
  });

  it("upsertAnnotation replaces an existing annotation by id, not duplicates it", () => {
    useAnnotationStore.getState().setAnnotations([makeAnnotation({ id: "a1", x1: 0.1 })]);
    useAnnotationStore.getState().upsertAnnotation(makeAnnotation({ id: "a1", x1: 0.5 }));
    const state = useAnnotationStore.getState();
    expect(state.annotations).toHaveLength(1);
    expect(state.annotations[0].x1).toBe(0.5);
  });

  it("removeAnnotation drops it from the list and clears selection if it was selected", () => {
    useAnnotationStore.getState().setAnnotations([makeAnnotation({ id: "a1" })]);
    useAnnotationStore.getState().selectAnnotation("a1");
    useAnnotationStore.getState().removeAnnotation("a1");
    const state = useAnnotationStore.getState();
    expect(state.annotations).toHaveLength(0);
    expect(state.selectedId).toBeNull();
  });

  it("removeAnnotation leaves selection alone if a different annotation was selected", () => {
    useAnnotationStore.getState().setAnnotations([makeAnnotation({ id: "a1" }), makeAnnotation({ id: "a2" })]);
    useAnnotationStore.getState().selectAnnotation("a2");
    useAnnotationStore.getState().removeAnnotation("a1");
    expect(useAnnotationStore.getState().selectedId).toBe("a2");
  });

  it("setMode('draw') clears the current selection", () => {
    useAnnotationStore.getState().selectAnnotation("a1");
    useAnnotationStore.getState().setMode("draw");
    expect(useAnnotationStore.getState().selectedId).toBeNull();
  });

  it("setMode('select') preserves the current selection", () => {
    useAnnotationStore.getState().selectAnnotation("a1");
    useAnnotationStore.getState().setMode("select");
    expect(useAnnotationStore.getState().selectedId).toBe("a1");
  });
});
