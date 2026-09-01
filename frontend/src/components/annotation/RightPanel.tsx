import { useState } from "react";
import type { Annotation, AnnotationFlag, ClassEntry } from "@/types";
import { classColor } from "@/config/classColors";

interface Props {
  annotation: Annotation | null;
  annotations: Annotation[];
  classEntries: ClassEntry[];
  flags: AnnotationFlag[];
  onChangeClass: (classId: number, className: string) => void;
  onEditCoords: (patch: Pick<Annotation, "x1" | "y1" | "x2" | "y2">) => void;
  onDelete: () => void;
  onDuplicate: () => void;
  onResolveFlag: (flagId: string, resolution: "CONFIRMED_FP" | "CONFIRMED_OK") => void;
  // The "active label" for the NEXT box you draw — separate from
  // `annotation.class_id` above, which relabels a box that already
  // exists. Same split CVAT/Roboflow make: picking here doesn't touch
  // whatever's currently selected.
  drawClassId: number;
  onSelectDrawClass: (classId: number) => void;
  onAddClass: (name: string) => void;
  addingClass: boolean;
  pendingShape: boolean;
  onCancelPending: () => void;
  onDeleteImage: () => void;
  deletingImage: boolean;
  collapsed: boolean;
  onToggleCollapse: () => void;
}

function DeleteImageSection({ onDeleteImage, deletingImage }: { onDeleteImage: () => void; deletingImage: boolean }) {
  const [confirming, setConfirming] = useState(false);

  if (confirming) {
    return (
      <div className="mt-6 border-2 border-accent p-3">
        <p className="mb-2 text-[10px] font-bold uppercase tracking-widest text-accent-ink">
          Delete this image and its annotations?
        </p>
        <div className="flex gap-2">
          <button
            onClick={onDeleteImage}
            disabled={deletingImage}
            className="flex-1 border-2 border-accent bg-accent py-1.5 text-[10px] font-bold uppercase tracking-widest text-paper hover:border-orange hover:bg-orange hover:text-ink disabled:opacity-40"
          >
            {deletingImage ? "Deleting…" : "Delete permanently"}
          </button>
          <button
            onClick={() => setConfirming(false)}
            className="border-2 border-ink/30 px-3 py-1.5 text-[10px] font-bold uppercase tracking-widest hover:border-orange hover:bg-orange hover:text-ink"
          >
            Cancel
          </button>
        </div>
      </div>
    );
  }

  return (
    <button
      onClick={() => setConfirming(true)}
      className="mt-6 border-2 border-ink/20 py-2 text-[10px] font-bold uppercase tracking-widest text-ink/60 transition-colors duration-150 hover:border-orange hover:text-ink"
    >
      Delete image
    </button>
  );
}

function ClassPicker({
  classEntries,
  drawClassId,
  onSelectDrawClass,
  onAddClass,
  addingClass,
  pendingShape,
  onCancelPending,
}: {
  classEntries: ClassEntry[];
  drawClassId: number;
  onSelectDrawClass: (classId: number) => void;
  onAddClass: (name: string) => void;
  addingClass: boolean;
  // When a box has just been drawn, this panel switches from "pick the
  // default for the NEXT box" to "classify THIS box" — same chip list,
  // different consequence for clicking one (see AnnotatePage).
  pendingShape: boolean;
  onCancelPending: () => void;
}) {
  const [adding, setAdding] = useState(false);
  const [newName, setNewName] = useState("");

  function submit() {
    const name = newName.trim();
    if (!name) return;
    // A name that only differs by case or stray whitespace from an existing
    // class reads as "the same class" to a human but would otherwise mint a
    // second class_config entry with a new id — so annotations end up split
    // across two "identical-looking" classes. Reuse the existing one instead.
    const existing = classEntries.find((c) => c.name.trim().toLowerCase() === name.toLowerCase());
    if (existing) {
      onSelectDrawClass(existing.id);
    } else {
      onAddClass(name);
    }
    setNewName("");
    setAdding(false);
  }

  return (
    <div className={`mb-6 border-b-2 pb-6 ${pendingShape ? "border-[#FFB000]" : "border-ink"}`}>
      <p
        className={`mb-2 text-[10px] font-bold uppercase tracking-widest ${
          pendingShape ? "text-[#FFB000]" : "text-ink/60"
        }`}
      >
        {pendingShape ? "New shape — pick a class" : "Drawing as — next shape uses this class"}
      </p>
      <div className="flex flex-wrap gap-1.5">
        {classEntries.map((c, i) => (
          <button
            key={c.id}
            onClick={() => onSelectDrawClass(c.id)}
            className={`flex items-center gap-1.5 border-2 px-2 py-1 text-xs font-semibold uppercase tracking-wide transition-colors duration-150 ${
              !pendingShape && c.id === drawClassId
                ? "border-ink bg-ink text-paper"
                : "border-ink/20 hover:border-ink"
            }`}
          >
            <span className="h-2 w-2 shrink-0" style={{ backgroundColor: classColor(i) }} />
            {c.name}
            {i < 9 && <span className="text-[9px] font-normal opacity-50">{i + 1}</span>}
          </button>
        ))}
        {classEntries.length === 0 && (
          <p className="text-xs text-ink/60">
            {pendingShape ? "No classes yet — add one below to classify this box." : "No classes yet — add one below to start drawing."}
          </p>
        )}
      </div>
      {pendingShape && (
        <button
          onClick={onCancelPending}
          className="mt-2 text-[10px] font-bold uppercase tracking-widest text-ink/60 underline decoration-1 underline-offset-2 hover:text-ink"
        >
          Cancel — discard this box
        </button>
      )}

      {adding ? (
        <div className="mt-2 flex gap-1.5">
          <input
            autoFocus
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") submit();
              if (e.key === "Escape") setAdding(false);
            }}
            aria-label="NEW CLASS NAME"
            placeholder="NEW CLASS NAME"
            className="min-w-0 flex-1 border-2 border-ink bg-paper px-2 py-1 text-xs outline-none focus:border-accent"
          />
          <button
            onClick={submit}
            disabled={!newName.trim() || addingClass}
            className="border-2 border-ink bg-ink px-3 text-xs font-bold uppercase tracking-widest text-paper disabled:opacity-40"
          >
            {addingClass ? "…" : "Add"}
          </button>
        </div>
      ) : (
        <button
          onClick={() => setAdding(true)}
          className="mt-2 text-[10px] font-bold uppercase tracking-widest text-ink/60 underline decoration-1 underline-offset-2 hover:text-ink"
        >
          + Add new class
        </button>
      )}
    </div>
  );
}

// One glance answer to "what's actually in this image" — independent of
// whatever box happens to be selected, so it stays visible whether or not
// anything is picked (both RightPanel branches below render it). Grouped
// straight from the annotations' own class_id/class_name rather than joined
// against the project's class_config: that config can lag what's actually
// on an image (e.g. an AUTO annotation from a model whose classes were
// never set as the project's taxonomy), and this should show what's really
// there regardless — same fallback color-by-class_id pattern as "Selected"
// below uses when a class isn't found in classEntries.
function ClassCounts({ annotations, classEntries }: { annotations: Annotation[]; classEntries: ClassEntry[] }) {
  if (annotations.length === 0) return null;

  const byClass = new Map<number, { name: string; count: number }>();
  for (const a of annotations) {
    const existing = byClass.get(a.class_id);
    if (existing) existing.count += 1;
    else byClass.set(a.class_id, { name: a.class_name, count: 1 });
  }

  return (
    <div className="mb-6 border-b-2 border-ink pb-6">
      <p className="mb-2 text-[10px] font-bold uppercase tracking-widest text-ink/60">
        In this image — {annotations.length} box{annotations.length === 1 ? "" : "es"}
      </p>
      <div className="flex flex-wrap gap-1.5">
        {[...byClass.entries()].map(([classId, { name, count }]) => {
          const colorIndex = classEntries.findIndex((c) => c.id === classId);
          return (
            <span
              key={classId}
              className="flex items-center gap-1.5 border border-ink/20 px-2 py-1 text-xs font-semibold uppercase tracking-wide"
            >
              <span
                className="h-2 w-2 shrink-0"
                style={{ backgroundColor: classColor(colorIndex >= 0 ? colorIndex : classId) }}
              />
              {name}
              <span className="tabular text-ink/60">{count}</span>
            </span>
          );
        })}
      </div>
    </div>
  );
}

const SOURCE_LABEL: Record<string, string> = { AUTO: "Auto", HUMAN: "Human", CORRECTED: "Corrected" };

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between border-b border-ink/10 py-2">
      <span className="text-[10px] font-bold uppercase tracking-widest text-ink/60">{label}</span>
      {children}
    </div>
  );
}

function CoordInput({ value, onCommit }: { value: number; onCommit: (v: number) => void }) {
  return (
    <input
      type="number"
      step={0.001}
      min={0}
      max={1}
      defaultValue={value.toFixed(3)}
      key={value} // resync when external value changes (e.g. after a drag)
      onBlur={(e) => {
        const v = parseFloat(e.target.value);
        if (!Number.isNaN(v)) onCommit(Math.min(1, Math.max(0, v)));
      }}
      className="tabular w-20 border border-ink/20 bg-paper px-1.5 py-0.5 text-right text-xs outline-none focus:border-accent"
    />
  );
}

function CollapseTab({ collapsed, onToggle }: { collapsed: boolean; onToggle: () => void }) {
  return (
    <button
      onClick={onToggle}
      aria-label={collapsed ? "Expand annotation panel" : "Collapse annotation panel"}
      className="flex h-8 w-8 shrink-0 items-center justify-center self-end border-b-2 border-l-2 border-ink text-xs font-bold hover:bg-orange hover:text-ink"
    >
      {collapsed ? "‹" : "›"}
    </button>
  );
}

export function RightPanel({
  annotation,
  annotations,
  classEntries,
  flags,
  onChangeClass,
  onEditCoords,
  onDelete,
  onDuplicate,
  onResolveFlag,
  drawClassId,
  onSelectDrawClass,
  onAddClass,
  addingClass,
  pendingShape,
  onCancelPending,
  onDeleteImage,
  deletingImage,
  collapsed,
  onToggleCollapse,
}: Props) {
  if (collapsed) {
    return (
      <aside className="flex h-full w-8 shrink-0 flex-col border-l-4 border-ink bg-paper">
        <CollapseTab collapsed={collapsed} onToggle={onToggleCollapse} />
      </aside>
    );
  }

  if (!annotation) {
    return (
      <aside className="flex h-full w-72 shrink-0 flex-col overflow-y-auto border-l-4 border-ink bg-paper">
        <CollapseTab collapsed={collapsed} onToggle={onToggleCollapse} />
        <div className="flex-1 p-6">
          <ClassPicker
            classEntries={classEntries}
            drawClassId={drawClassId}
            onSelectDrawClass={onSelectDrawClass}
            onAddClass={onAddClass}
            addingClass={addingClass}
            pendingShape={pendingShape}
            onCancelPending={onCancelPending}
          />
          <ClassCounts annotations={annotations} classEntries={classEntries} />
          <p className="text-xs font-bold uppercase tracking-widest text-ink/60">
            No annotation selected
          </p>
          <DeleteImageSection onDeleteImage={onDeleteImage} deletingImage={deletingImage} />
        </div>
      </aside>
    );
  }

  const colorIndex = classEntries.findIndex((c) => c.id === annotation.class_id);
  const color = classColor(colorIndex >= 0 ? colorIndex : annotation.class_id);

  return (
    <aside className="flex h-full w-72 shrink-0 flex-col overflow-y-auto border-l-4 border-ink bg-paper">
      <CollapseTab collapsed={collapsed} onToggle={onToggleCollapse} />
      <div className="flex-1 p-6">
      <ClassPicker
        classEntries={classEntries}
        drawClassId={drawClassId}
        onSelectDrawClass={onSelectDrawClass}
        onAddClass={onAddClass}
        addingClass={addingClass}
        pendingShape={pendingShape}
        onCancelPending={onCancelPending}
      />
      <ClassCounts annotations={annotations} classEntries={classEntries} />

      <div className="mb-4 flex items-center gap-2">
        <span className="h-3 w-3 shrink-0" style={{ backgroundColor: color }} />
        <h3 className="text-sm font-bold uppercase tracking-widest">Selected</h3>
      </div>

      <Field label="Class">
        <select
          value={annotation.class_id}
          onChange={(e) => {
            const entry = classEntries.find((c) => c.id === Number(e.target.value));
            if (entry) onChangeClass(entry.id, entry.name);
          }}
          className="border border-ink/20 bg-paper px-1.5 py-1 text-xs font-semibold uppercase outline-none focus:border-accent"
        >
          {classEntries.map((c) => (
            <option key={c.id} value={c.id}>
              {c.name}
            </option>
          ))}
        </select>
      </Field>

      <Field label="Confidence">
        <span className="tabular text-xs">
          {annotation.confidence != null ? annotation.confidence.toFixed(3) : "—"}
        </span>
      </Field>

      {annotation.shape_type === "BBOX" ? (
        <>
          <Field label="X1">
            <CoordInput value={annotation.x1} onCommit={(v) => onEditCoords({ ...annotation, x1: v })} />
          </Field>
          <Field label="Y1">
            <CoordInput value={annotation.y1} onCommit={(v) => onEditCoords({ ...annotation, y1: v })} />
          </Field>
          <Field label="X2">
            <CoordInput value={annotation.x2} onCommit={(v) => onEditCoords({ ...annotation, x2: v })} />
          </Field>
          <Field label="Y2">
            <CoordInput value={annotation.y2} onCommit={(v) => onEditCoords({ ...annotation, y2: v })} />
          </Field>
        </>
      ) : (
        // A polygon's geometry is edited by dragging its vertices on the
        // canvas, not by typing numbers — numeric bbox editing doesn't mean
        // anything for a shape with more than 4 degrees of freedom.
        <Field label="Points">
          <span className="tabular text-xs">{annotation.points?.length ?? 0}</span>
        </Field>
      )}

      <Field label="Source">
        <span
          className={`px-2 py-0.5 text-[10px] font-bold uppercase tracking-widest ${
            annotation.source === "AUTO" ? "bg-muted text-ink/70" : "bg-ink text-paper"
          }`}
        >
          {SOURCE_LABEL[annotation.source]}
        </span>
      </Field>

      {flags.length > 0 && (
        <div className="mt-6 border-2 border-accent">
          <div className="border-b-2 border-accent bg-accent px-3 py-1.5">
            <p className="text-[10px] font-bold uppercase tracking-widest text-paper">
              ⚠ {flags.length} flag{flags.length === 1 ? "" : "s"}
            </p>
          </div>
          <div className="divide-y divide-accent/20">
            {flags.map((flag) => (
              <div key={flag.id} className="p-3">
                <p className="mb-1 text-[10px] font-bold uppercase tracking-widest text-accent-ink">
                  {flag.flag_type.replace(/_/g, " ")}
                </p>
                <p className="mb-2 text-xs text-ink/70">{flag.reason}</p>
                {flag.resolution ? (
                  <span className="text-[10px] font-bold uppercase tracking-widest text-ink/60">
                    Resolved: {flag.resolution}
                  </span>
                ) : (
                  <div className="flex gap-2">
                    <button
                      onClick={() => onResolveFlag(flag.id, "CONFIRMED_FP")}
                      className="border border-ink px-2 py-1 text-[10px] font-bold uppercase tracking-widest hover:border-orange hover:bg-orange hover:text-ink"
                    >
                      False positive
                    </button>
                    <button
                      onClick={() => onResolveFlag(flag.id, "CONFIRMED_OK")}
                      className="border border-ink px-2 py-1 text-[10px] font-bold uppercase tracking-widest hover:border-orange hover:bg-orange hover:text-ink"
                    >
                      Looks fine
                    </button>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="mt-6 flex flex-col gap-2">
        <button
          onClick={onDuplicate}
          className="border-2 border-ink py-2 text-xs font-bold uppercase tracking-widest transition-colors duration-150 hover:border-orange hover:bg-orange hover:text-ink"
        >
          Duplicate
        </button>
        <button
          onClick={onDelete}
          className="border-2 border-ink bg-paper py-2 text-xs font-bold uppercase tracking-widest text-ink transition-colors duration-150 hover:bg-orange hover:text-ink hover:border-orange"
        >
          Delete
        </button>
      </div>

      <DeleteImageSection onDeleteImage={onDeleteImage} deletingImage={deletingImage} />
      </div>
    </aside>
  );
}
