import { useEffect, useRef, useState } from "react";
import type { Annotation, AnnotationFlag, ClassEntry } from "@/types";
import { classColor } from "@/config/classColors";
import { useAnnotationStore } from "@/store/annotationStore";

export interface CanvasControls {
  zoomIn: () => void;
  zoomOut: () => void;
  fit: () => void;
  actualSize: () => void;
}

interface Props {
  imageUrl: string;
  imageWidth: number;
  imageHeight: number;
  classEntries: ClassEntry[];
  onCommitMove: (id: string, patch: Pick<Annotation, "x1" | "y1" | "x2" | "y2">) => void;
  onCreateBox: (box: Pick<Annotation, "x1" | "y1" | "x2" | "y2">) => void;
  controlsRef: React.MutableRefObject<CanvasControls | null>;
  /** Unresolved flags, keyed by annotation id. Red is reserved exclusively
   * for this signal (DESIGN.md "Box colour language") — never a class
   * colour, always layered on top of the class's own stroke. */
  flagsByAnnotationId?: Record<string, AnnotationFlag[]>;
}

const HANDLES = ["nw", "n", "ne", "e", "se", "s", "sw", "w"] as const;
type Handle = (typeof HANDLES)[number];

function colorForClass(classId: number, classEntries: ClassEntry[]): string {
  const index = classEntries.findIndex((c) => c.id === classId);
  return classColor(index >= 0 ? index : classId);
}

/** Instrument-register annotation canvas: an <img> as the black-plate
 * ground, an SVG overlay in the same coordinate space carrying the boxes.
 * Pan/zoom is a single CSS transform on the shared wrapper (`stageRef`) —
 * box geometry never has to account for zoom, only screen-space pointer
 * math does (via getBoundingClientRect(), which already reflects the
 * current transform). See PLAN "Annotation canvas: SVG boxes over a
 * <canvas> image". */
export function AnnotationCanvas({
  imageUrl,
  imageWidth,
  imageHeight,
  classEntries,
  onCommitMove,
  onCreateBox,
  controlsRef,
  flagsByAnnotationId = {},
}: Props) {
  const wrapperRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const boxGroupRefs = useRef<Map<string, SVGGElement>>(new Map());
  const boxRectRefs = useRef<Map<string, SVGRectElement>>(new Map());
  const draftRectRef = useRef<SVGRectElement>(null);

  const annotations = useAnnotationStore((s) => s.annotations);
  const selectedId = useAnnotationStore((s) => s.selectedId);
  const mode = useAnnotationStore((s) => s.mode);
  const selectAnnotation = useAnnotationStore((s) => s.selectAnnotation);
  const setMode = useAnnotationStore((s) => s.setMode);

  const [scale, setScale] = useState(1);
  const transformRef = useRef({ scale: 1, tx: 0, ty: 0 });

  const applyTransform = () => {
    const { scale: s, tx, ty } = transformRef.current;
    if (stageRef.current) {
      stageRef.current.style.transform = `translate(${tx}px, ${ty}px) scale(${s})`;
    }
  };

  const fit = () => {
    const wrapper = wrapperRef.current;
    if (!wrapper) return;
    const padding = 32;
    const availW = wrapper.clientWidth - padding * 2;
    const availH = wrapper.clientHeight - padding * 2;
    const s = Math.min(availW / imageWidth, availH / imageHeight, 1);
    transformRef.current = {
      scale: s,
      tx: (wrapper.clientWidth - imageWidth * s) / 2,
      ty: (wrapper.clientHeight - imageHeight * s) / 2,
    };
    applyTransform();
    setScale(s);
  };

  const zoomBy = (factor: number) => {
    const wrapper = wrapperRef.current;
    if (!wrapper) return;
    const cx = wrapper.clientWidth / 2;
    const cy = wrapper.clientHeight / 2;
    const { scale: s, tx, ty } = transformRef.current;
    const newScale = Math.min(8, Math.max(0.05, s * factor));
    // keep the viewport centre point fixed while scaling
    const imgX = (cx - tx) / s;
    const imgY = (cy - ty) / s;
    transformRef.current = {
      scale: newScale,
      tx: cx - imgX * newScale,
      ty: cy - imgY * newScale,
    };
    applyTransform();
    setScale(newScale);
  };

  const actualSize = () => {
    const wrapper = wrapperRef.current;
    if (!wrapper) return;
    transformRef.current = {
      scale: 1,
      tx: (wrapper.clientWidth - imageWidth) / 2,
      ty: (wrapper.clientHeight - imageHeight) / 2,
    };
    applyTransform();
    setScale(1);
  };

  useEffect(() => {
    controlsRef.current = { zoomIn: () => zoomBy(1.25), zoomOut: () => zoomBy(0.8), fit, actualSize };
  });

  // Fit on first load and whenever a new image's dimensions arrive.
  useEffect(() => {
    fit();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [imageWidth, imageHeight, imageUrl]);

  function stagePointFromEvent(e: { clientX: number; clientY: number }): { x: number; y: number } {
    const rect = svgRef.current!.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * imageWidth;
    const y = ((e.clientY - rect.top) / rect.height) * imageHeight;
    return { x: Math.min(Math.max(x, 0), imageWidth), y: Math.min(Math.max(y, 0), imageHeight) };
  }

  // --- Wheel zoom, centred on the cursor ---
  function handleWheel(e: React.WheelEvent) {
    e.preventDefault();
    const wrapper = wrapperRef.current;
    if (!wrapper) return;
    const rect = wrapper.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    const { scale: s, tx, ty } = transformRef.current;
    const factor = e.deltaY < 0 ? 1.1 : 0.9;
    const newScale = Math.min(8, Math.max(0.05, s * factor));
    const imgX = (cx - tx) / s;
    const imgY = (cy - ty) / s;
    transformRef.current = { scale: newScale, tx: cx - imgX * newScale, ty: cy - imgY * newScale };
    applyTransform();
    setScale(newScale);
  }

  // --- Pan (background drag in select mode) ---
  const panState = useRef<{ startX: number; startY: number; startTx: number; startTy: number } | null>(null);

  function handleBackgroundPointerDown(e: React.PointerEvent) {
    if (mode === "draw") {
      startDraw(e);
      return;
    }
    selectAnnotation(null);
    panState.current = {
      startX: e.clientX,
      startY: e.clientY,
      startTx: transformRef.current.tx,
      startTy: transformRef.current.ty,
    };
    (e.target as Element).setPointerCapture(e.pointerId);
  }

  function handleBackgroundPointerMove(e: React.PointerEvent) {
    if (panState.current) {
      const dx = e.clientX - panState.current.startX;
      const dy = e.clientY - panState.current.startY;
      transformRef.current.tx = panState.current.startTx + dx;
      transformRef.current.ty = panState.current.startTy + dy;
      applyTransform();
    } else if (drawState.current) {
      updateDraw(e);
    }
  }

  function handleBackgroundPointerUp(e: React.PointerEvent) {
    if (panState.current) {
      panState.current = null;
    } else if (drawState.current) {
      finishDraw(e);
    }
  }

  // --- Draw new box ---
  const drawState = useRef<{ startX: number; startY: number } | null>(null);
  const [draft, setDraft] = useState<{ x: number; y: number; w: number; h: number } | null>(null);

  function startDraw(e: React.PointerEvent) {
    const p = stagePointFromEvent(e);
    drawState.current = { startX: p.x, startY: p.y };
    setDraft({ x: p.x, y: p.y, w: 0, h: 0 });
    (e.target as Element).setPointerCapture(e.pointerId);
  }

  function updateDraw(e: React.PointerEvent) {
    if (!drawState.current || !draftRectRef.current) return;
    const p = stagePointFromEvent(e);
    const x = Math.min(drawState.current.startX, p.x);
    const y = Math.min(drawState.current.startY, p.y);
    const w = Math.abs(p.x - drawState.current.startX);
    const h = Math.abs(p.y - drawState.current.startY);
    draftRectRef.current.setAttribute("x", String(x));
    draftRectRef.current.setAttribute("y", String(y));
    draftRectRef.current.setAttribute("width", String(w));
    draftRectRef.current.setAttribute("height", String(h));
  }

  function finishDraw(e: React.PointerEvent) {
    if (!drawState.current) return;
    const p = stagePointFromEvent(e);
    const x1 = Math.min(drawState.current.startX, p.x);
    const y1 = Math.min(drawState.current.startY, p.y);
    const x2 = Math.max(drawState.current.startX, p.x);
    const y2 = Math.max(drawState.current.startY, p.y);
    drawState.current = null;
    setDraft(null);
    setMode("select");
    const MIN_PX = 4;
    if (x2 - x1 < MIN_PX || y2 - y1 < MIN_PX) return; // treat as an accidental click, not a box
    onCreateBox({
      x1: x1 / imageWidth,
      y1: y1 / imageHeight,
      x2: x2 / imageWidth,
      y2: y2 / imageHeight,
    });
  }

  // --- Move / resize an existing box ---
  const dragState = useRef<{
    annotationId: string;
    kind: "move" | "resize";
    handle?: Handle;
    startPointer: { x: number; y: number };
    startBox: { x1: number; y1: number; x2: number; y2: number };
  } | null>(null);
  const liveBoxRef = useRef<{ x1: number; y1: number; x2: number; y2: number } | null>(null);

  function beginBoxDrag(e: React.PointerEvent, annotation: Annotation, handle?: Handle) {
    e.stopPropagation();
    selectAnnotation(annotation.id);
    const p = stagePointFromEvent(e);
    dragState.current = {
      annotationId: annotation.id,
      kind: handle ? "resize" : "move",
      handle,
      startPointer: p,
      startBox: {
        x1: annotation.x1 * imageWidth,
        y1: annotation.y1 * imageHeight,
        x2: annotation.x2 * imageWidth,
        y2: annotation.y2 * imageHeight,
      },
    };
    (e.target as Element).setPointerCapture(e.pointerId);
  }

  function updateBoxDrag(e: React.PointerEvent) {
    const drag = dragState.current;
    if (!drag) return;
    const p = stagePointFromEvent(e);
    const dx = p.x - drag.startPointer.x;
    const dy = p.y - drag.startPointer.y;
    let { x1, y1, x2, y2 } = drag.startBox;

    if (drag.kind === "move") {
      const w = x2 - x1;
      const h = y2 - y1;
      x1 = Math.min(Math.max(x1 + dx, 0), imageWidth - w);
      y1 = Math.min(Math.max(y1 + dy, 0), imageHeight - h);
      x2 = x1 + w;
      y2 = y1 + h;
    } else if (drag.handle) {
      if (drag.handle.includes("w")) x1 = Math.min(Math.max(x1 + dx, 0), x2 - 2);
      if (drag.handle.includes("e")) x2 = Math.max(Math.min(x2 + dx, imageWidth), x1 + 2);
      if (drag.handle.includes("n")) y1 = Math.min(Math.max(y1 + dy, 0), y2 - 2);
      if (drag.handle.includes("s")) y2 = Math.max(Math.min(y2 + dy, imageHeight), y1 + 2);
    }

    const rect = boxRectRefs.current.get(drag.annotationId);
    if (rect) {
      rect.setAttribute("x", String(x1));
      rect.setAttribute("y", String(y1));
      rect.setAttribute("width", String(x2 - x1));
      rect.setAttribute("height", String(y2 - y1));
    }
    liveBoxRef.current = { x1, y1, x2, y2 };
  }

  function endBoxDrag() {
    const drag = dragState.current;
    const live = liveBoxRef.current;
    dragState.current = null;
    liveBoxRef.current = null;
    if (!drag || !live) return;
    onCommitMove(drag.annotationId, {
      x1: live.x1 / imageWidth,
      y1: live.y1 / imageHeight,
      x2: live.x2 / imageWidth,
      y2: live.y2 / imageHeight,
    });
  }

  function handleSvgPointerMove(e: React.PointerEvent) {
    if (dragState.current) {
      updateBoxDrag(e);
    } else {
      handleBackgroundPointerMove(e);
    }
  }

  function handleSvgPointerUp(e: React.PointerEvent) {
    if (dragState.current) {
      endBoxDrag();
    } else {
      handleBackgroundPointerUp(e);
    }
  }

  const HANDLE_SCREEN_PX = 7;
  const handleSize = HANDLE_SCREEN_PX / scale;

  function handlePosition(handle: Handle, x1: number, y1: number, x2: number, y2: number) {
    const midX = (x1 + x2) / 2;
    const midY = (y1 + y2) / 2;
    const map: Record<Handle, [number, number]> = {
      nw: [x1, y1],
      n: [midX, y1],
      ne: [x2, y1],
      e: [x2, midY],
      se: [x2, y2],
      s: [midX, y2],
      sw: [x1, y2],
      w: [x1, midY],
    };
    return map[handle];
  }

  function cursorForHandle(handle: Handle): string {
    const map: Record<Handle, string> = {
      nw: "nwse-resize",
      se: "nwse-resize",
      ne: "nesw-resize",
      sw: "nesw-resize",
      n: "ns-resize",
      s: "ns-resize",
      e: "ew-resize",
      w: "ew-resize",
    };
    return map[handle];
  }

  return (
    <div
      ref={wrapperRef}
      className="relative h-full w-full overflow-hidden bg-plate"
      onWheel={handleWheel}
    >
      <div
        ref={stageRef}
        className="absolute left-0 top-0 origin-top-left"
        style={{ width: imageWidth, height: imageHeight }}
      >
        {/* max-w-none defeats Tailwind preflight's `img { max-width: 100% }`
            reset, which would otherwise clamp this to the wrapper's CSS
            width and silently desync it from the SVG overlay's native-pixel
            coordinate space (the SVG isn't subject to that reset). */}
        <img
          src={imageUrl}
          width={imageWidth}
          height={imageHeight}
          draggable={false}
          className="block max-w-none select-none"
          alt=""
        />
        <svg
          ref={svgRef}
          width={imageWidth}
          height={imageHeight}
          viewBox={`0 0 ${imageWidth} ${imageHeight}`}
          className="absolute left-0 top-0"
          style={{ cursor: mode === "draw" ? "crosshair" : "default" }}
          onPointerDown={handleBackgroundPointerDown}
          onPointerMove={handleSvgPointerMove}
          onPointerUp={handleSvgPointerUp}
        >
          {annotations.map((ann) => {
            const color = colorForClass(ann.class_id, classEntries);
            const selected = ann.id === selectedId;
            const boxW = (ann.x2 - ann.x1) * imageWidth;
            const boxH = (ann.y2 - ann.y1) * imageHeight;
            const x = ann.x1 * imageWidth;
            const y = ann.y1 * imageHeight;
            const flags = flagsByAnnotationId[ann.id]?.filter((f) => !f.resolution) ?? [];
            const isFlagged = flags.length > 0;
            return (
              <g
                key={ann.id}
                ref={(el) => {
                  if (el) boxGroupRefs.current.set(ann.id, el);
                  else boxGroupRefs.current.delete(ann.id);
                }}
              >
                <rect
                  ref={(el) => {
                    if (el) boxRectRefs.current.set(ann.id, el);
                    else boxRectRefs.current.delete(ann.id);
                  }}
                  x={x}
                  y={y}
                  width={boxW}
                  height={boxH}
                  fill="transparent"
                  stroke={color}
                  strokeWidth={selected ? 3 : 2}
                  strokeDasharray={ann.source === "AUTO" ? "6 3" : undefined}
                  vectorEffect="non-scaling-stroke"
                  style={{ cursor: mode === "select" ? "move" : "default" }}
                  onPointerDown={(e) => mode === "select" && beginBoxDrag(e, ann)}
                />
                {/* Red is reserved exclusively for "needs attention" — layered
                    on top of the class's own colour, never replacing it
                    (DESIGN.md "Box colour language"). */}
                {isFlagged && (
                  <rect
                    x={x - 3 / scale}
                    y={y - 3 / scale}
                    width={boxW + 6 / scale}
                    height={boxH + 6 / scale}
                    fill="none"
                    stroke="#FF3000"
                    strokeWidth={2}
                    strokeDasharray="3 2"
                    vectorEffect="non-scaling-stroke"
                    pointerEvents="none"
                  />
                )}
                {/* Label tag welded to the top-left corner (DESIGN.md) */}
                <g transform={`translate(${x}, ${y - 16 / scale})`}>
                  <rect
                    x={0}
                    y={0}
                    width={Math.max(10, (ann.class_name.length + (isFlagged ? 2 : 0)) * (7 / scale) + 8 / scale)}
                    height={14 / scale}
                    fill={isFlagged ? "#FF3000" : color}
                  />
                  <text
                    x={4 / scale}
                    y={11 / scale}
                    fontSize={10 / scale}
                    fontWeight={700}
                    fill={isFlagged ? "#FFFFFF" : "#000"}
                    style={{ fontFamily: "Inter, sans-serif", textTransform: "uppercase" }}
                  >
                    {isFlagged ? "⚠ " : ""}
                    {ann.class_name}
                    {ann.confidence != null ? ` ${ann.confidence.toFixed(2)}` : ""}
                  </text>
                </g>
                {selected &&
                  mode === "select" &&
                  HANDLES.map((h) => {
                    const [hx, hy] = handlePosition(h, x, y, x + boxW, y + boxH);
                    return (
                      <rect
                        key={h}
                        x={hx - handleSize / 2}
                        y={hy - handleSize / 2}
                        width={handleSize}
                        height={handleSize}
                        fill="#FFFFFF"
                        stroke="#000000"
                        strokeWidth={1.5}
                        vectorEffect="non-scaling-stroke"
                        style={{ cursor: cursorForHandle(h) }}
                        onPointerDown={(e) => beginBoxDrag(e, ann, h)}
                      />
                    );
                  })}
              </g>
            );
          })}
          {draft && (
            <rect
              ref={draftRectRef}
              x={draft.x}
              y={draft.y}
              width={draft.w}
              height={draft.h}
              fill="rgba(255,255,255,0.1)"
              stroke="#FFFFFF"
              strokeWidth={2}
              strokeDasharray="4 4"
              vectorEffect="non-scaling-stroke"
            />
          )}
        </svg>
      </div>
    </div>
  );
}
