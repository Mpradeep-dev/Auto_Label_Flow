import { SHORTCUTS } from "@/hooks/shortcuts";

export function ShortcutHelp({ onClose }: { onClose: () => void }) {
  return (
    <div
      className="absolute inset-0 z-20 flex items-center justify-center bg-ink/60"
      onClick={onClose}
    >
      <div
        className="w-80 border-4 border-ink bg-paper p-6"
        onClick={(e) => e.stopPropagation()}
      >
        <h3 className="mb-4 text-sm font-bold uppercase tracking-widest">Keyboard shortcuts</h3>
        <ul>
          {SHORTCUTS.map((s) => (
            <li key={s.id} className="flex items-center justify-between border-b border-ink/10 py-1.5">
              <span className="text-xs text-ink/70">{s.label}</span>
              <kbd className="border border-ink/30 bg-muted px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-widest">
                {s.key}
              </kbd>
            </li>
          ))}
        </ul>
        <button
          onClick={onClose}
          className="mt-4 w-full border-2 border-ink py-2 text-xs font-bold uppercase tracking-widest hover:border-orange hover:bg-orange hover:text-paper"
        >
          Close
        </button>
      </div>
    </div>
  );
}
