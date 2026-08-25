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

  const imagesQuery = useQuery({
    queryKey: ["images", datasetId],
    queryFn: () => api.listImages(datasetId!, 200, 0),
    enabled: !!datasetId,
  });
  const images = imagesQuery.data?.items ?? [];
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

  const createMutation = useMutation({
    mutationFn: (box: Pick<Annotation, "x1" | "y1" | "x2" | "y2">) => {
      // No silent "unknown" fallback — the auto-select effect above keeps
      // `drawClassId` pointed at a real class the moment one exists, so
      // this only actually throws in the edge case of a project with zero
      // classes (no detector registered yet) where someone drew anyway.
      const cls = classEntries.find((c) => c.id === drawClassId);
      if (!cls) throw new Error("No class selected — pick or add one in the right panel before drawing.");
      return api.createAnnotation({
        image_id: imageId!,
        class_id: cls.id,
        class_name: cls.name,
        ...box,
      });
    },
    onSuccess: (ann) => {
      upsertAnnotation(ann);
      selectAnnotation(ann.id);
    },
  });

  const addClassMutation = useMutation({
    mutationFn: (name: string) => {
      const nextId = classEntries.length > 0 ? Math.max(...classEntries.map((c) => c.id)) + 1 : 0;
      return api.updateProject(projectId!, { class_config: [...classEntries, { id: nextId, name }] });
    },
    onSuccess: (project) => {
      queryClient.setQueryData(["project", projectId], project);
      const created = project.class_config[project.class_config.length - 1];
      if (created) setDrawClassId(created.id);
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
        if (mode === "draw") setDrawClassId(cls.id);
        else if (selected) updateMutation.mutate({ id: selected.id, patch: { class_id: cls.id, class_name: cls.name } });
      },
    },
    !!imageId,
  );

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
            onCreateBox={(box) => createMutation.mutate(box)}
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
          onSelectDrawClass={setDrawClassId}
          onAddClass={(name) => addClassMutation.mutate(name)}
          addingClass={addClassMutation.isPending}
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
