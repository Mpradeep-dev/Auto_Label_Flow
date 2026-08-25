import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/services/api";
import { AnnotationCanvas, type CanvasControls } from "@/components/annotation/AnnotationCanvas";
import { RightPanel } from "@/components/annotation/RightPanel";
import { Toolbar } from "@/components/annotation/Toolbar";
import { Filmstrip } from "@/components/annotation/Filmstrip";
import { ShortcutHelp } from "@/components/annotation/ShortcutHelp";
import { useAnnotationStore } from "@/store/annotationStore";
import { useKeyboardShortcuts } from "@/hooks/useKeyboardShortcuts";
import type { Annotation, AnnotationFlag } from "@/types";

export function AnnotatePage() {
  const { projectId, datasetId, imageId } = useParams<{
    projectId: string;
    datasetId: string;
    imageId: string;
  }>();
  const navigate = useNavigate();
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

  const setAnnotations = useAnnotationStore((s) => s.setAnnotations);
  const annotations = useAnnotationStore((s) => s.annotations);
  const selectedId = useAnnotationStore((s) => s.selectedId);
  const selectAnnotation = useAnnotationStore((s) => s.selectAnnotation);
  const mode = useAnnotationStore((s) => s.mode);
  const setMode = useAnnotationStore((s) => s.setMode);
  const drawClassId = useAnnotationStore((s) => s.drawClassId);
  const setDrawClassId = useAnnotationStore((s) => s.setDrawClassId);
  const pendingBox = useAnnotationStore((s) => s.pendingBox);
  const setPendingBox = useAnnotationStore((s) => s.setPendingBox);
  const upsertAnnotation = useAnnotationStore((s) => s.upsertAnnotation);
  const removeAnnotationLocal = useAnnotationStore((s) => s.removeAnnotation);

  const projectQuery = useQuery({
    queryKey: ["project", projectId],
    queryFn: () => api.getProject(projectId!),
    enabled: !!projectId,
  });
  const classEntries = projectQuery.data?.class_config ?? [];

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
  const currentIndex = images.findIndex((i) => i.id === imageId);
  const currentImage = currentIndex >= 0 ? images[currentIndex] : undefined;
  const prevImage = currentIndex > 0 ? images[currentIndex - 1] : undefined;
  const nextImage = currentIndex >= 0 && currentIndex < images.length - 1 ? images[currentIndex + 1] : undefined;

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

  function goTo(id: string | undefined) {
    if (!id) return;
    navigate(`/projects/${projectId}/datasets/${datasetId}/images/${id}/annotate`);
  }

  // Class choice happens AFTER a box is drawn (see `pendingBox`): drawing
  // just stages geometry, and this is the one place that turns a pending
  // box + a chosen class into a real, persisted annotation.
  const createMutation = useMutation({
    mutationFn: ({
      box,
      classId,
      className,
    }: {
      box: Pick<Annotation, "x1" | "y1" | "x2" | "y2">;
      classId: number;
      className: string;
    }) =>
      api.createAnnotation({
        image_id: imageId!,
        class_id: classId,
        class_name: className,
        ...box,
      }),
    onSuccess: (ann) => {
      upsertAnnotation(ann);
      selectAnnotation(ann.id);
      setPendingBox(null);
      setDrawClassId(ann.class_id); // pre-highlight the same class for next time
    },
  });

  // Picking a class in the right panel means one of two things depending on
  // whether a box is waiting to be classified: with a pendingBox, it
  // commits that box as a real annotation; otherwise it just sets the
  // highlighted default for whenever the next box gets drawn.
  function pickClass(classId: number) {
    const cls = classEntries.find((c) => c.id === classId);
    if (!cls) return;
    if (pendingBox) createMutation.mutate({ box: pendingBox, classId: cls.id, className: cls.name });
    else setDrawClassId(classId);
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
      if (pendingBox) createMutation.mutate({ box: pendingBox, classId: match.id, className: match.name });
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
    },
  });

  const approveMutation = useMutation({
    mutationFn: () => api.approveImage(imageId!),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["images", datasetId] });
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
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["images", datasetId] });
      queryClient.invalidateQueries({ queryKey: ["review-queue"] });
    },
  });

  const deleteImageMutation = useMutation({
    mutationFn: () => api.deleteImage(imageId!),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["images", datasetId] });
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

  function handleSave() {
    queryClient.invalidateQueries({ queryKey: ["annotations", imageId] });
    setJustSaved(true);
    setTimeout(() => setJustSaved(false), 1200);
  }

  useKeyboardShortcuts(
    {
      add: () => setMode(mode === "draw" ? "select" : "draw"),
      delete: () => selected && deleteMutation.mutate(selected.id),
      edit: () => {}, // right panel coordinate fields are always live-editable; nothing to focus-toggle
      prev: () => goTo(prevImage?.id),
      next: () => goTo(nextImage?.id),
      approve: () => approveMutation.mutate(),
      save: handleSave,
      zoom: () => controlsRef.current?.zoomIn(),
      fit: () => controlsRef.current?.fit(),
      setClassByIndex: (index) => {
        const cls = classEntries[index];
        if (!cls) return;
        // A pendingBox takes priority: a digit key is the fast path to
        // classify the box that's actually waiting right now.
        if (pendingBox) createMutation.mutate({ box: pendingBox, classId: cls.id, className: cls.name });
        else if (selected) updateMutation.mutate({ id: selected.id, patch: { class_id: cls.id, class_name: cls.name } });
        else setDrawClassId(cls.id);
      },
    },
    !!imageId,
  );

  // Escape discards a pending box instead of persisting it — the one
  // keyboard shortcut here that isn't in the shared declarative map, since
  // it's specific to this transient staging state, not a global action.
  useEffect(() => {
    if (!pendingBox) return;
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") setPendingBox(null);
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [pendingBox, setPendingBox]);

  if (!projectId || !datasetId || !imageId) return null;

  if (imagesQuery.isLoading || annotationsQuery.isLoading || !currentImage) {
    return <div className="flex h-full items-center justify-center text-sm text-ink/40">Loading…</div>;
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
            onCommitMove={(id, patch) => updateMutation.mutate({ id, patch })}
            onBoxDrawn={(box) => setPendingBox(box)}
            pendingBox={pendingBox}
            flagsByAnnotationId={flagsByAnnotationId}
          />
          {showShortcuts && <ShortcutHelp onClose={() => setShowShortcuts(false)} />}
          {createMutation.isError && (
            <p className="absolute left-4 top-4 border-2 border-accent bg-paper px-3 py-2 text-xs font-bold text-accent">
              {(createMutation.error as Error).message}
            </p>
          )}
        </div>
        <RightPanel
          annotation={selected}
          annotations={annotations}
          classEntries={classEntries}
          flags={selected ? (flagsByAnnotationId[selected.id] ?? []) : []}
          onChangeClass={(classId, className) =>
            selected && updateMutation.mutate({ id: selected.id, patch: { class_id: classId, class_name: className } })
          }
          onEditCoords={(patch) => selected && updateMutation.mutate({ id: selected.id, patch })}
          onDelete={() => selected && deleteMutation.mutate(selected.id)}
          onDuplicate={() => selected && duplicateMutation.mutate(selected.id)}
          onResolveFlag={(flagId, resolution) => resolveFlagMutation.mutate({ flagId, resolution })}
          drawClassId={drawClassId}
          onSelectDrawClass={pickClass}
          onAddClass={(name) => addClassMutation.mutate(name)}
          addingClass={addClassMutation.isPending}
          pendingBox={!!pendingBox}
          onCancelPending={() => setPendingBox(null)}
          onDeleteImage={() => deleteImageMutation.mutate()}
          deletingImage={deleteImageMutation.isPending}
          collapsed={rightPanelCollapsed}
          onToggleCollapse={() => setRightPanelCollapsed((v) => !v)}
        />
      </div>
      <Filmstrip images={images} currentId={imageId} onSelect={goTo} />
      <Toolbar
        position={`${currentIndex + 1} / ${images.length}`}
        hasSelection={!!selected}
        drawing={mode === "draw"}
        reviewStatus={currentImage.review_status}
        approving={approveMutation.isPending}
        rejecting={rejectMutation.isPending}
        approveError={approveMutation.isError ? (approveMutation.error as Error).message || "Approve failed" : null}
        rejectError={rejectMutation.isError ? (rejectMutation.error as Error).message || "Reject failed" : null}
        onPrev={() => goTo(prevImage?.id)}
        onNext={() => goTo(nextImage?.id)}
        onApprove={() => approveMutation.mutate()}
        onReject={() => rejectMutation.mutate()}
        onSave={handleSave}
        justSaved={justSaved}
        onDeleteSelected={() => selected && deleteMutation.mutate(selected.id)}
        onAdd={() => setMode(mode === "draw" ? "select" : "draw")}
        onZoomIn={() => controlsRef.current?.zoomIn()}
        onZoomOut={() => controlsRef.current?.zoomOut()}
        onFit={() => controlsRef.current?.fit()}
        onToggleShortcuts={() => setShowShortcuts((v) => !v)}
      />
    </div>
  );
}
