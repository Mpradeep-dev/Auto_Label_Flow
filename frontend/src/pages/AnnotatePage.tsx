import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useParams, useSearchParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/services/api";
import { AnnotationCanvas, type CanvasControls } from "@/components/annotation/AnnotationCanvas";
import { RightPanel } from "@/components/annotation/RightPanel";
import { Toolbar } from "@/components/annotation/Toolbar";
import { Filmstrip } from "@/components/annotation/Filmstrip";
import { ShortcutHelp } from "@/components/annotation/ShortcutHelp";
import { useAnnotationStore } from "@/store/annotationStore";
import type { PendingShape } from "@/store/annotationStore";
import { useKeyboardShortcuts } from "@/hooks/useKeyboardShortcuts";
import type { Annotation, AnnotationFlag, AnnotationImage } from "@/types";

// Audit finding FE-02: there was no undo anywhere in the canvas — every
// create/move/resize/reclass/delete persisted immediately with no history,
// so a misclick had no recovery but manually redoing the work. A single
// stack of the last N edits, reversed one at a time with Ctrl/Cmd+Z: a
// created box is undone by deleting it, a delete by recreating it (new id
// — undo doesn't need to be byte-identical, just to put the geometry/class
// back), and a move/resize/reclass by reapplying the pre-edit field values.
type UndoEntry =
  | { type: "create"; annotationId: string }
  | { type: "delete"; annotation: Annotation }
  | { type: "update"; id: string; previous: Partial<Annotation> };

const UNDO_STACK_LIMIT = 25;

export function AnnotatePage() {
  const { projectId, datasetId, imageId } = useParams<{
    projectId: string;
    datasetId: string;
    imageId: string;
  }>();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const queryClient = useQueryClient();
  const controlsRef = useRef<CanvasControls | null>(null);
  const [showShortcuts, setShowShortcuts] = useState(false);
  // Laptop-width screens default to collapsed — the fixed 288px panel eats
  // too much of the canvas below ~1024px. Wider screens keep it open.
  const [rightPanelCollapsed, setRightPanelCollapsed] = useState(
    () => typeof window !== "undefined" && window.innerWidth < 1024,
  );
  // Every annotation edit (create/move/resize/reclass/delete) already
  // persists immediately via its own mutation — Save has nothing left to
  // do but refresh from the server, so without this flash it silently
  // succeeds at doing effectively nothing, which reads as "the button is
  // broken" rather than "there was nothing to save."
  const [justSaved, setJustSaved] = useState(false);
  // Set briefly whenever navigation/approve/reject is blocked because a
  // drawn shape hasn't been classified yet — see `guardPendingShape` below.
  // Fixes the silent-data-loss bug where leaving the image used to discard
  // an unclassified shape with zero warning (audit finding FE-01).
  const [pendingShapeWarning, setPendingShapeWarning] = useState(false);
  // The last SAM prompt point, while its segment request is in flight —
  // cleared on success (pendingShape takes over) or failure. Local state,
  // not the annotation store: it's this page's own request lifecycle, not
  // canvas interaction state other components need to read.
  const [samPending, setSamPending] = useState<{ x: number; y: number } | null>(null);
  // Ref, not state — undo doesn't need a re-render on push, only on the
  // rare undo() call itself, and a ref avoids fighting the mutations'
  // own state updates within the same tick.
  const undoStackRef = useRef<UndoEntry[]>([]);
  // Set true for the duration of an undo's own mutate() call so its
  // success handler doesn't push a fresh undo entry for the undo itself
  // (which would make Ctrl+Z immediately re-doable as "undo the undo"
  // rather than actually going back further).
  const isUndoingRef = useRef(false);

  useEffect(() => {
    undoStackRef.current = [];
  }, [imageId]);

  function pushUndo(entry: UndoEntry) {
    if (isUndoingRef.current) return;
    undoStackRef.current.push(entry);
    if (undoStackRef.current.length > UNDO_STACK_LIMIT) undoStackRef.current.shift();
  }

  const setAnnotations = useAnnotationStore((s) => s.setAnnotations);
  const annotations = useAnnotationStore((s) => s.annotations);
  const selectedId = useAnnotationStore((s) => s.selectedId);
  const selectAnnotation = useAnnotationStore((s) => s.selectAnnotation);
  const mode = useAnnotationStore((s) => s.mode);
  const setMode = useAnnotationStore((s) => s.setMode);
  const drawClassId = useAnnotationStore((s) => s.drawClassId);
  const setDrawClassId = useAnnotationStore((s) => s.setDrawClassId);
  const pendingShape = useAnnotationStore((s) => s.pendingShape);
  const setPendingShape = useAnnotationStore((s) => s.setPendingShape);
  const upsertAnnotation = useAnnotationStore((s) => s.upsertAnnotation);
  const removeAnnotationLocal = useAnnotationStore((s) => s.removeAnnotation);

  const projectQuery = useQuery({
    queryKey: ["project", projectId],
    queryFn: () => api.getProject(projectId!),
    enabled: !!projectId,
  });
  const classEntries = projectQuery.data?.class_config ?? [];

  // Which SAM checkpoint (if any) to use for interactive segmentation —
  // installed status lives on the Settings page, this just reads it.
  // "sam-full" is preferred when both are installed (better mask quality).
  const samModelsQuery = useQuery({ queryKey: ["system-sam-models"], queryFn: () => api.listSamModels() });
  const installedSamVariant = samModelsQuery.data?.find((m) => m.name === "sam-full" && m.installed)
    ? "sam-full"
    : samModelsQuery.data?.find((m) => m.name === "sam-lite" && m.installed)
      ? "sam-lite"
      : null;

  // `drawClassId` defaults to 0 in the store, which only accidentally means
  // something once real classes exist (see createMutation below) — once
  // they load, snap it to an actual class so a fresh page load can never
  // silently draw as "unknown" just because nobody picked a class yet.
  useEffect(() => {
    if (classEntries.length > 0 && !classEntries.some((c) => c.id === drawClassId)) {
      setDrawClassId(classEntries[0].id);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [classEntries]);

  // Needs the full, correctly-ordered dataset — not one page of it — since
  // prev/next/approve-and-advance and the filmstrip all walk this array.
  // A hardcoded single-page fetch here used to silently cap navigation at
  // the server's 200-per-request limit: on any dataset bigger than that,
  // Next/Approve could never reach images past #200, and opening one
  // directly (e.g. from the Review Queue, sorted by difficulty rather than
  // upload order) could land on an image outside that first page entirely.
  const imagesQuery = useQuery({
    queryKey: ["images", datasetId, "all"],
    queryFn: () => api.listAllImages(datasetId!),
    enabled: !!datasetId,
  });
  const images = imagesQuery.data ?? [];

  // Arrived from the Review Queue (?from=review-queue&reviewStatus=...) —
  // that page browses PENDING images by difficulty_score DESC (or APPROVED
  // by most-recently-approved), not the dataset's upload order `images`
  // above is in. Without re-deriving that same order here, Prev/Next/the
  // filmstrip would silently jump to plain upload order the moment you
  // opened an image, which reads as "the review order changed" even though
  // nothing on the server did.
  const fromReviewQueue = searchParams.get("from") === "review-queue";
  // Arrived from one of ImagesPage's own Pending/Approved/Rejected tabs
  // (?from=images&reviewStatus=...) — same problem, plainer order: without
  // this, opening an image from e.g. the Pending tab and hitting Next would
  // silently start walking the FULL, unfiltered dataset instead of staying
  // inside that tab's filtered set.
  const fromImagesTab = searchParams.get("from") === "images";
  const reviewStatusParam = searchParams.get("reviewStatus") as "PENDING" | "APPROVED" | "REJECTED" | null;
  const flagTypeParam = searchParams.get("flagType");
  const reviewOrderQuery = useQuery({
    queryKey: ["review-queue-order", projectId, datasetId, reviewStatusParam, flagTypeParam],
    queryFn: () =>
      api.listAllReviewQueueImageIds({
        project_id: projectId!,
        dataset_id: datasetId,
        review_status: reviewStatusParam ?? undefined,
        flag_type: flagTypeParam ?? undefined,
      }),
    enabled: fromReviewQueue && !!projectId && !!datasetId,
  });
  const imagesTabQuery = useQuery({
    queryKey: ["images", datasetId, "all", reviewStatusParam],
    queryFn: () => api.listAllImages(datasetId!, reviewStatusParam ?? undefined),
    enabled: fromImagesTab && !!datasetId && !!reviewStatusParam,
  });

  const orderedImages = useMemo(() => {
    if (fromReviewQueue && reviewOrderQuery.data) {
      const byId = new Map(images.map((img) => [img.id, img]));
      return reviewOrderQuery.data.map((id) => byId.get(id)).filter((img): img is (typeof images)[number] => !!img);
    }
    if (fromImagesTab && imagesTabQuery.data) return imagesTabQuery.data;
    return images;
  }, [fromReviewQueue, reviewOrderQuery.data, fromImagesTab, imagesTabQuery.data, images]);

  const currentIndex = orderedImages.findIndex((i) => i.id === imageId);
  // Falls back to `images` if the current image isn't in the (still
  // loading, or filtered-differently) review order yet — otherwise the
  // very image you clicked into could momentarily fail to render.
  const currentImage =
    currentIndex >= 0 ? orderedImages[currentIndex] : images.find((i) => i.id === imageId);
  const prevImage = currentIndex > 0 ? orderedImages[currentIndex - 1] : undefined;
  const nextImage =
    currentIndex >= 0 && currentIndex < orderedImages.length - 1 ? orderedImages[currentIndex + 1] : undefined;

  const annotationsQuery = useQuery({
    queryKey: ["annotations", imageId],
    queryFn: () => api.listAnnotations(imageId!),
    enabled: !!imageId,
  });
  const flagsQuery = useQuery({
    queryKey: ["flags", imageId],
    queryFn: () => api.listImageFlags(imageId!),
    enabled: !!imageId,
  });
  const flagsByAnnotationId = (flagsQuery.data ?? []).reduce<Record<string, AnnotationFlag[]>>(
    (acc, flag) => {
      (acc[flag.annotation_id] ??= []).push(flag);
      return acc;
    },
    {},
  );

  useEffect(() => {
    if (annotationsQuery.data) setAnnotations(annotationsQuery.data);
  }, [annotationsQuery.data, setAnnotations]);

  // Prefetch neighbours: instant "Next"/"Prev" with no request-shaped wait.
  useEffect(() => {
    if (prevImage) {
      queryClient.prefetchQuery({ queryKey: ["annotations", prevImage.id], queryFn: () => api.listAnnotations(prevImage.id) });
      const img = new Image();
      img.src = prevImage.url;
    }
    if (nextImage) {
      queryClient.prefetchQuery({ queryKey: ["annotations", nextImage.id], queryFn: () => api.listAnnotations(nextImage.id) });
      const img = new Image();
      img.src = nextImage.url;
    }
  }, [prevImage, nextImage, queryClient]);

  // Returns true if it's safe to leave the current shape state (navigate,
  // approve, reject, delete-image). Returns false — and flashes a warning
  // instead — if a shape has been drawn but not yet classified, so that
  // state can never be silently dropped: the user must either pick a class
  // (RightPanel's class chips) or explicitly discard it (Cancel button /
  // Escape) before anything here proceeds.
  function guardPendingShape(): boolean {
    if (!pendingShape) return true;
    setPendingShapeWarning(true);
    setTimeout(() => setPendingShapeWarning(false), 2600);
    return false;
  }

  function goTo(id: string | undefined) {
    if (!id) return;
    if (!guardPendingShape()) return;
    // Preserve ?from=review-queue&... so stepping through Next/Prev/the
    // filmstrip keeps browsing in review-queue order instead of reverting
    // to dataset upload order after the first navigation.
    const qs = searchParams.toString();
    navigate(`/projects/${projectId}/datasets/${datasetId}/images/${id}/annotate${qs ? `?${qs}` : ""}`);
  }

  function shapeToCreatePayload(shape: PendingShape, classId: number, className: string) {
    return shape.shape_type === "BBOX"
      ? {
          image_id: imageId!,
          class_id: classId,
          class_name: className,
          shape_type: "BBOX" as const,
          x1: shape.x1,
          y1: shape.y1,
          x2: shape.x2,
          y2: shape.y2,
        }
      : {
          image_id: imageId!,
          class_id: classId,
          class_name: className,
          shape_type: "POLYGON" as const,
          points: shape.points,
        };
  }

  // Class choice happens AFTER a shape is drawn (see `pendingShape`):
  // drawing just stages geometry, and this is the one place that turns a
  // pending shape + a chosen class into a real, persisted annotation.
  const createMutation = useMutation({
    mutationFn: ({
      shape,
      classId,
      className,
    }: {
      shape: PendingShape;
      classId: number;
      className: string;
    }) => api.createAnnotation(shapeToCreatePayload(shape, classId, className)),
    onSuccess: (ann) => {
      upsertAnnotation(ann);
      selectAnnotation(ann.id);
      setPendingShape(null);
      setDrawClassId(ann.class_id); // pre-highlight the same class for next time
      pushUndo({ type: "create", annotationId: ann.id });
    },
  });

  // A SAM prompt point in, a traced polygon out — same "geometry only, class
  // choice happens after" contract onBoxDrawn/onPolygonDrawn already use:
  // success stages the result as pendingShape rather than creating anything.
  const samSegmentMutation = useMutation({
    mutationFn: (point: { x: number; y: number }) =>
      api.segmentImage(imageId!, { variant: installedSamVariant!, points: [[point.x, point.y]] }),
    onSuccess: (result) => {
      setSamPending(null);
      if (result.points) setPendingShape({ shape_type: "POLYGON", points: result.points });
      // else: SAM found nothing usable — samSegmentMutation.isError stays
      // false (this wasn't a failure), the click marker just disappears
      // and the tool stays active for another attempt.
    },
    onError: () => setSamPending(null),
  });

  function handleSamPoint(point: { x: number; y: number }) {
    if (pendingShape || samPending || !installedSamVariant) return;
    setSamPending(point);
    samSegmentMutation.mutate(point);
  }

  // Picking a class in the right panel means one of two things depending on
  // whether a shape is waiting to be classified: with a pendingShape, it
  // commits that shape as a real annotation; otherwise it just sets the
  // highlighted default for whenever the next shape gets drawn.
  function pickClass(classId: number) {
    const cls = classEntries.find((c) => c.id === classId);
    if (!cls) return;
    // Guard against a double-click firing two POSTs for the same pending
    // shape (audit finding FE-09) — once the first create is in flight
    // there is no pendingShape left to double-submit against a moment
    // later, but the in-flight window itself needs an explicit guard.
    if (pendingShape && !createMutation.isPending) createMutation.mutate({ shape: pendingShape, classId: cls.id, className: cls.name });
    else if (!pendingShape) setDrawClassId(classId);
  }

  const addClassMutation = useMutation({
    mutationFn: (name: string) => {
      // Defense in depth against ClassPicker's own case/whitespace check:
      // never persist a class_config entry that duplicates an existing name.
      const existing = classEntries.find((c) => c.name.trim().toLowerCase() === name.toLowerCase());
      if (existing) return Promise.resolve(projectQuery.data!);
      const nextId = classEntries.length > 0 ? Math.max(...classEntries.map((c) => c.id)) + 1 : 0;
      return api.updateProject(projectId!, { class_config: [...classEntries, { id: nextId, name }] });
    },
    onSuccess: (project, name) => {
      queryClient.setQueryData(["project", projectId], project);
      const match = project.class_config.find((c) => c.name.trim().toLowerCase() === name.toLowerCase());
      if (!match) return;
      if (pendingShape) createMutation.mutate({ shape: pendingShape, classId: match.id, className: match.name });
      else setDrawClassId(match.id);
    },
  });

  const updateMutation = useMutation({
    mutationFn: ({ id, patch }: { id: string; patch: Partial<Annotation> }) => api.updateAnnotation(id, patch),
    onSuccess: (ann) => upsertAnnotation(ann),
  });

  const deleteMutation = useMutation({
    mutationFn: (id: string) => api.deleteAnnotation(id),
    onSuccess: (_void, id) => removeAnnotationLocal(id),
  });

  const duplicateMutation = useMutation({
    mutationFn: (id: string) => api.duplicateAnnotation(id),
    onSuccess: (ann) => {
      upsertAnnotation(ann);
      selectAnnotation(ann.id);
      pushUndo({ type: "create", annotationId: ann.id });
    },
  });

  // updateMutation/deleteMutation only get the record *after* the change —
  // useless for undo, which needs the pre-edit state. These wrappers
  // capture it from current local state before the mutation fires, and
  // only commit it to the undo stack once the mutation actually succeeds
  // (a failed edit shouldn't leave a stale undo entry for a change that
  // never happened).
  function commitUpdate(id: string, patch: Partial<Annotation>) {
    const current = annotations.find((a) => a.id === id);
    const previous: Partial<Annotation> = {};
    if (current) {
      for (const key of Object.keys(patch) as (keyof Annotation)[]) {
        (previous as Record<string, unknown>)[key] = current[key];
      }
    }
    updateMutation.mutate(
      { id, patch },
      { onSuccess: () => current && pushUndo({ type: "update", id, previous }) },
    );
  }

  function commitDelete(id: string) {
    const current = annotations.find((a) => a.id === id);
    deleteMutation.mutate(id, {
      onSuccess: () => current && pushUndo({ type: "delete", annotation: current }),
    });
  }

  function undo() {
    const entry = undoStackRef.current.pop();
    if (!entry) return;
    isUndoingRef.current = true;
    const release = () => {
      isUndoingRef.current = false;
    };
    if (entry.type === "create") {
      deleteMutation.mutate(entry.annotationId, { onSettled: release });
    } else if (entry.type === "delete") {
      const a = entry.annotation;
      const shape: PendingShape =
        a.shape_type === "POLYGON" && a.points
          ? { shape_type: "POLYGON", points: a.points }
          : { shape_type: "BBOX", x1: a.x1, y1: a.y1, x2: a.x2, y2: a.y2 };
      createMutation.mutate(
        { shape, classId: a.class_id, className: a.class_name },
        { onSettled: release },
      );
    } else {
      updateMutation.mutate({ id: entry.id, patch: entry.previous }, { onSettled: release });
    }
  }

  // Patches the "all images" cache AnnotatePage itself keeps (`imagesQuery`
  // below, key ["images", datasetId, "all"]) in place, instead of
  // invalidating it. On a large dataset `listAllImages` pages through the
  // whole dataset (thousands of images, many round trips even parallelized)
  // — approve/reject/delete only ever change ONE image's row, so
  // invalidating that whole cached list here used to force a full dataset
  // refetch on every single click, which is what made stepping through a
  // big dataset feel slow. `refetchType: "none"` still marks the broader
  // ["images", datasetId] prefix (e.g. ImagesPage's own paginated cache)
  // stale so *those* pages pick up the change next time they're visited,
  // without forcing an immediate refetch of anything right now.
  function patchCachedImage(updated: AnnotationImage) {
    queryClient.setQueryData<AnnotationImage[]>(["images", datasetId, "all"], (old) =>
      old?.map((img) => (img.id === updated.id ? updated : img)),
    );
    // Browsing a status-filtered tab (?from=images&reviewStatus=...): once
    // this image's status no longer matches that tab, it needs to actually
    // drop out of the cached list here — not just have its field patched —
    // or it keeps showing up in e.g. the Pending tab's Prev/Next/Filmstrip
    // after being approved (the literal bug report this fixes).
    if (fromImagesTab && reviewStatusParam) {
      queryClient.setQueryData<AnnotationImage[]>(["images", datasetId, "all", reviewStatusParam], (old) =>
        updated.review_status === reviewStatusParam
          ? old?.map((img) => (img.id === updated.id ? updated : img))
          : old?.filter((img) => img.id !== updated.id),
      );
    }
    queryClient.invalidateQueries({ queryKey: ["images", datasetId], refetchType: "none" });
  }

  const approveMutation = useMutation({
    mutationFn: () => api.approveImage(imageId!),
    onSuccess: (updatedImage) => {
      patchCachedImage(updatedImage);
      queryClient.invalidateQueries({ queryKey: ["annotations", imageId] });
      // The Review Queue's PENDING/APPROVED tabs cache for 30s (default
      // staleTime) — without this, approving here and switching straight
      // back to that page can still show the image sitting in PENDING.
      queryClient.invalidateQueries({ queryKey: ["review-queue"] });
      // Reviewing is a loop — approve, see the next one, approve, ... —
      // so advance automatically instead of leaving the reviewer parked on
      // an image that's now done. Stays put if this was the last one
      // (`goTo` no-ops on an undefined id).
      goTo(nextImage?.id);
    },
  });
  const rejectMutation = useMutation({
    mutationFn: () => api.rejectImage(imageId!),
    onSuccess: (updatedImage) => {
      patchCachedImage(updatedImage);
      queryClient.invalidateQueries({ queryKey: ["review-queue"] });
    },
  });

  const deleteImageMutation = useMutation({
    mutationFn: () => api.deleteImage(imageId!),
    onSuccess: () => {
      queryClient.setQueryData<AnnotationImage[]>(["images", datasetId, "all"], (old) =>
        old?.filter((img) => img.id !== imageId),
      );
      if (fromImagesTab && reviewStatusParam) {
        queryClient.setQueryData<AnnotationImage[]>(["images", datasetId, "all", reviewStatusParam], (old) =>
          old?.filter((img) => img.id !== imageId),
        );
      }
      queryClient.invalidateQueries({ queryKey: ["images", datasetId], refetchType: "none" });
      queryClient.invalidateQueries({ queryKey: ["dataset-stats", datasetId] });
      // Land on whatever's next in the sequence rather than a 404 on the
      // image we just removed; if this was the only/last image, there's
      // nothing left to annotate — back out to the list.
      const target = nextImage ?? prevImage;
      if (target) goTo(target.id);
      else navigate(`/projects/${projectId}/datasets/${datasetId}/images`);
    },
  });

  const resolveFlagMutation = useMutation({
    mutationFn: ({ flagId, resolution }: { flagId: string; resolution: "CONFIRMED_FP" | "CONFIRMED_OK" }) =>
      api.resolveFlag(flagId, resolution),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["flags", imageId] }),
  });

  const selected = useMemo(() => annotations.find((a) => a.id === selectedId) ?? null, [annotations, selectedId]);

  function handleApprove() {
    if (!guardPendingShape()) return;
    approveMutation.mutate();
  }
  function handleReject() {
    if (!guardPendingShape()) return;
    rejectMutation.mutate();
  }
  function handleDeleteImage() {
    if (!guardPendingShape()) return;
    deleteImageMutation.mutate();
  }

  function handleSave() {
    // Every real edit already persists immediately via its own mutation —
    // the one thing NOT yet persisted is a pendingShape, so flashing
    // "Saved" while one exists would be a false signal (audit finding FE-10).
    if (!guardPendingShape()) return;
    queryClient.invalidateQueries({ queryKey: ["annotations", imageId] });
    setJustSaved(true);
    setTimeout(() => setJustSaved(false), 1200);
  }

  useKeyboardShortcuts(
    {
      drawBbox: () => setMode(mode === "draw-bbox" ? "select" : "draw-bbox"),
      drawPolygon: () => setMode(mode === "draw-polygon" ? "select" : "draw-polygon"),
      drawSam: () => installedSamVariant && setMode(mode === "draw-sam" ? "select" : "draw-sam"),
      delete: () => selected && commitDelete(selected.id),
      undo,
      prev: () => goTo(prevImage?.id),
      next: () => goTo(nextImage?.id),
      approve: handleApprove,
      save: handleSave,
      zoom: () => controlsRef.current?.zoomIn(),
      zoomOut: () => controlsRef.current?.zoomOut(),
      fit: () => controlsRef.current?.fit(),
      setClassByIndex: (index) => {
        const cls = classEntries[index];
        if (!cls) return;
        // A pendingShape takes priority: a digit key is the fast path to
        // classify the shape that's actually waiting right now.
        if (pendingShape) createMutation.mutate({ shape: pendingShape, classId: cls.id, className: cls.name });
        else if (selected) commitUpdate(selected.id, { class_id: cls.id, class_name: cls.name });
        else setDrawClassId(cls.id);
      },
    },
    !!imageId,
  );

  // Escape discards a pending shape instead of persisting it — the one
  // keyboard shortcut here that isn't in the shared
  // declarative map, since it's specific to this transient staging state,
  // not a global action. (An in-progress, not-yet-closed polygon ring has
  // its own, earlier Escape handling inside AnnotationCanvas — this only
  // fires once a shape has actually finished drawing and is awaiting a
  // class pick.)
  useEffect(() => {
    if (!pendingShape) return;
    function onKeyDown(e: KeyboardEvent) {
      if (e.key !== "Escape") return;
      setPendingShape(null);
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [pendingShape, setPendingShape]);

  if (!projectId || !datasetId || !imageId) return null;

  if (imagesQuery.isLoading || annotationsQuery.isLoading || !currentImage) {
    return <div className="flex h-full items-center justify-center text-sm text-ink/60">Loading…</div>;
  }

  return (
    <div className="flex h-full flex-col">
      <div className="flex min-h-0 flex-1">
        <div className="relative min-w-0 flex-1">
          <AnnotationCanvas
            imageUrl={currentImage.url}
            imageWidth={currentImage.width}
            imageHeight={currentImage.height}
            classEntries={classEntries}
            controlsRef={controlsRef}
            onCommitMove={(id, patch) => commitUpdate(id, patch)}
            onCommitPolygonMove={(id, points) => commitUpdate(id, { points })}
            onBoxDrawn={(box) => setPendingShape({ shape_type: "BBOX", ...box })}
            onPolygonDrawn={(points) => setPendingShape({ shape_type: "POLYGON", points })}
            onSamPoint={handleSamPoint}
            samPending={samPending}
            pendingShape={pendingShape}
            flagsByAnnotationId={flagsByAnnotationId}
          />
          {showShortcuts && <ShortcutHelp onClose={() => setShowShortcuts(false)} />}
          {pendingShapeWarning && (
            <p className="absolute left-4 top-4 border-2 border-accent bg-paper px-3 py-2 text-xs font-bold text-accent-ink">
              Pick a class for the shape you just drew, or press Esc to discard it, before moving on.
            </p>
          )}
          {createMutation.isError && (
            <p className="absolute left-4 top-4 border-2 border-accent bg-paper px-3 py-2 text-xs font-bold text-accent-ink">
              {(createMutation.error as Error).message}
            </p>
          )}
          {samSegmentMutation.isError && (
            <p className="absolute left-4 top-4 border-2 border-accent bg-paper px-3 py-2 text-xs font-bold text-accent-ink">
              {(samSegmentMutation.error as Error).message}
            </p>
          )}
        </div>
        <RightPanel
          annotation={selected}
          annotations={annotations}
          classEntries={classEntries}
          flags={selected ? (flagsByAnnotationId[selected.id] ?? []) : []}
          onChangeClass={(classId, className) =>
            selected && commitUpdate(selected.id, { class_id: classId, class_name: className })
          }
          onEditCoords={(patch) => selected && commitUpdate(selected.id, patch)}
          onDelete={() => selected && commitDelete(selected.id)}
          onDuplicate={() => selected && duplicateMutation.mutate(selected.id)}
          onResolveFlag={(flagId, resolution) => resolveFlagMutation.mutate({ flagId, resolution })}
          drawClassId={drawClassId}
          onSelectDrawClass={pickClass}
          onAddClass={(name) => addClassMutation.mutate(name)}
          addingClass={addClassMutation.isPending}
          pendingShape={!!pendingShape}
          onCancelPending={() => setPendingShape(null)}
          onDeleteImage={handleDeleteImage}
          deletingImage={deleteImageMutation.isPending}
          collapsed={rightPanelCollapsed}
          onToggleCollapse={() => setRightPanelCollapsed((v) => !v)}
        />
      </div>
      <Filmstrip images={orderedImages} currentId={imageId} onSelect={goTo} />
      <Toolbar
        position={`${currentIndex + 1} / ${orderedImages.length}`}
        hasSelection={!!selected}
        activeTool={
          mode === "draw-bbox" ? "bbox" : mode === "draw-polygon" ? "polygon" : mode === "draw-sam" ? "sam" : "select"
        }
        reviewStatus={currentImage.review_status}
        approving={approveMutation.isPending}
        rejecting={rejectMutation.isPending}
        approveError={approveMutation.isError ? (approveMutation.error as Error).message || "Approve failed" : null}
        rejectError={rejectMutation.isError ? (rejectMutation.error as Error).message || "Reject failed" : null}
        onPrev={() => goTo(prevImage?.id)}
        onNext={() => goTo(nextImage?.id)}
        onApprove={handleApprove}
        onReject={handleReject}
        onSave={handleSave}
        justSaved={justSaved}
        onDeleteSelected={() => selected && commitDelete(selected.id)}
        onSelectBboxTool={() => setMode(mode === "draw-bbox" ? "select" : "draw-bbox")}
        onSelectPolygonTool={() => setMode(mode === "draw-polygon" ? "select" : "draw-polygon")}
        onSelectSamTool={() => installedSamVariant && setMode(mode === "draw-sam" ? "select" : "draw-sam")}
        samDisabledReason={installedSamVariant ? null : "Download a SAM model in Settings → Desktop app first"}
        onZoomIn={() => controlsRef.current?.zoomIn()}
        onZoomOut={() => controlsRef.current?.zoomOut()}
        onFit={() => controlsRef.current?.fit()}
        onToggleShortcuts={() => setShowShortcuts((v) => !v)}
      />
    </div>
  );
}
