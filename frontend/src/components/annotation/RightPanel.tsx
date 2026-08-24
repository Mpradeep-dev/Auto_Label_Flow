import type { Annotation, AnnotationFlag, ClassEntry } from "@/types";
import { classColor } from "@/config/classColors";

interface Props {
  annotation: Annotation | null;
  classEntries: ClassEntry[];
  flags: AnnotationFlag[];
  onChangeClass: (classId: number, className: string) => void;
  onEditCoords: (patch: Pick<Annotation, "x1" | "y1" | "x2" | "y2">) => void;
  onDelete: () => void;
  onDuplicate: () => void;
  onResolveFlag: (flagId: string, resolution: "CONFIRMED_FP" | "CONFIRMED_OK") => void;
}

const SOURCE_LABEL: Record<string, string> = { AUTO: "Auto", HUMAN: "Human", CORRECTED: "Corrected" };

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between border-b border-ink/10 py-2">
      <span className="text-[10px] font-bold uppercase tracking-widest text-ink/50">{label}</span>
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

export function RightPanel({
  annotation,
  classEntries,
  flags,
  onChangeClass,
  onEditCoords,
  onDelete,
  onDuplicate,
  onResolveFlag,
}: Props) {
  if (!annotation) {
    return (
      <aside className="flex h-full w-72 shrink-0 flex-col border-l-4 border-ink bg-paper p-6">
        <p className="text-xs font-bold uppercase tracking-widest text-ink/40">
          No annotation selected
        </p>
      </aside>
    );
  }

  const colorIndex = classEntries.findIndex((c) => c.id === annotation.class_id);
  const color = classColor(colorIndex >= 0 ? colorIndex : annotation.class_id);

  return (
    <aside className="flex h-full w-72 shrink-0 flex-col overflow-y-auto border-l-4 border-ink bg-paper p-6">
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
                <p className="mb-1 text-[10px] font-bold uppercase tracking-widest text-accent">
                  {flag.flag_type.replace(/_/g, " ")}
                </p>
                <p className="mb-2 text-xs text-ink/70">{flag.reason}</p>
                {flag.resolution ? (
                  <span className="text-[10px] font-bold uppercase tracking-widest text-ink/40">
                    Resolved: {flag.resolution}
                  </span>
                ) : (
                  <div className="flex gap-2">
                    <button
                      onClick={() => onResolveFlag(flag.id, "CONFIRMED_FP")}
                      className="border border-ink px-2 py-1 text-[10px] font-bold uppercase tracking-widest hover:bg-ink hover:text-paper"
                    >
                      False positive
                    </button>
                    <button
                      onClick={() => onResolveFlag(flag.id, "CONFIRMED_OK")}
                      className="border border-ink px-2 py-1 text-[10px] font-bold uppercase tracking-widest hover:bg-ink hover:text-paper"
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
          className="border-2 border-ink py-2 text-xs font-bold uppercase tracking-widest transition-colors duration-150 hover:bg-ink hover:text-paper"
        >
          Duplicate
        </button>
        <button
          onClick={onDelete}
          className="border-2 border-ink bg-paper py-2 text-xs font-bold uppercase tracking-widest text-ink transition-colors duration-150 hover:bg-accent hover:text-paper hover:border-accent"
        >
          Delete
        </button>
      </div>
    </aside>
  );
}
