import { render, fireEvent } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { AnnotationCanvas, type CanvasControls } from "./AnnotationCanvas";
import { useAnnotationStore } from "@/store/annotationStore";
import type { PendingShape } from "@/store/annotationStore";

// jsdom's getBoundingClientRect returns all-zero by default, which would
// make the canvas's screen->image coordinate math (division by rect.width/
// height) produce NaN — stand in a fixed size matching the test image.
//
// jsdom also has no PointerEvent class at all (window.PointerEvent is
// undefined here), so testing-library's fireEvent.pointerDown(el, {clientX,
// clientY}) silently falls back to a plain Event with neither property set
// — every coordinate downstream would read as NaN. Dispatching a MouseEvent
// literally typed "pointerdown" instead works: React's onPointerDown just
// matches by the native event's `type` string, and MouseEvent (unlike
// PointerEvent) is one jsdom actually implements clientX/clientY on.
function pointerDownAt(el: Element, clientX: number, clientY: number) {
  fireEvent(el, new MouseEvent("pointerdown", { clientX, clientY, bubbles: true }));
}

beforeEach(() => {
  vi.spyOn(Element.prototype, "getBoundingClientRect").mockReturnValue({
    left: 0,
    top: 0,
    width: 200,
    height: 100,
    right: 200,
    bottom: 100,
    x: 0,
    y: 0,
    toJSON: () => "",
  });
  useAnnotationStore.setState({
    annotations: [],
    selectedId: null,
    mode: "draw-sam",
    drawClassId: 0,
    pendingShape: null,
  });
});

function renderCanvas(props: {
  onSamPoint?: (point: { x: number; y: number }) => void;
  samPending?: { x: number; y: number } | null;
  pendingShape?: PendingShape | null;
} = {}) {
  const controlsRef: React.MutableRefObject<CanvasControls | null> = { current: null };
  return render(
    <AnnotationCanvas
      imageUrl="http://example.test/img.jpg"
      imageWidth={200}
      imageHeight={100}
      classEntries={[]}
      onCommitMove={() => {}}
      onCommitPolygonMove={() => {}}
      onBoxDrawn={() => {}}
      onPolygonDrawn={() => {}}
      pendingShape={props.pendingShape ?? null}
      controlsRef={controlsRef}
      onSamPoint={props.onSamPoint}
      samPending={props.samPending}
    />,
  );
}

describe("AnnotationCanvas — draw-sam mode", () => {
  it("reports a normalized point on click", () => {
    const onSamPoint = vi.fn();
    const { container } = renderCanvas({ onSamPoint });

    const svg = container.querySelector("svg")!;
    pointerDownAt(svg, 100, 50); // centre of the 200x100 rect

    expect(onSamPoint).toHaveBeenCalledTimes(1);
    expect(onSamPoint).toHaveBeenCalledWith({ x: 0.5, y: 0.5 });
  });

  it("does not fire while a shape is already pending", () => {
    const onSamPoint = vi.fn();
    renderCanvas({
      onSamPoint,
      pendingShape: { shape_type: "POLYGON", points: [[0.1, 0.1], [0.2, 0.1], [0.2, 0.2]] },
    });
    const svg = document.querySelector("svg")!;
    pointerDownAt(svg, 100, 50);
    expect(onSamPoint).not.toHaveBeenCalled();
  });

  it("does not fire while a request is already in flight (samPending set)", () => {
    const onSamPoint = vi.fn();
    renderCanvas({ onSamPoint, samPending: { x: 0.2, y: 0.2 } });
    const svg = document.querySelector("svg")!;
    pointerDownAt(svg, 100, 50);
    expect(onSamPoint).not.toHaveBeenCalled();
  });
});
