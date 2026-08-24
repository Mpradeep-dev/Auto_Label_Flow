interface Props {
  onPrev: () => void;
  onNext: () => void;
  onApprove: () => void;
  onReject: () => void;
  onSave: () => void;
  onDeleteSelected: () => void;
  onAdd: () => void;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onFit: () => void;
  onToggleShortcuts: () => void;
  hasSelection: boolean;
  drawing: boolean;
  position: string; // "N / total"
}

function ToolButton({
  label,
  shortcut,
  onClick,
  disabled,
  active,
}: {
  label: string;
  shortcut?: string;
  onClick: () => void;
  disabled?: boolean;
  active?: boolean;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`flex h-full items-center gap-1.5 border-r border-ink/20 px-3 text-xs font-bold uppercase tracking-widest transition-colors duration-150 ${
        active ? "bg-ink text-paper" : "hover:bg-muted"
      } disabled:cursor-not-allowed disabled:opacity-30`}
    >
      {label}
      {shortcut && <span className="text-[9px] font-normal text-ink/40">{shortcut}</span>}
    </button>
  );
}

export function Toolbar(props: Props) {
  return (
    <div className="flex h-11 shrink-0 items-stretch border-t-4 border-ink bg-paper">
      <ToolButton label="← Prev" shortcut="←" onClick={props.onPrev} />
      <ToolButton label="Next →" shortcut="→" onClick={props.onNext} />
      <div className="flex items-center border-r border-ink/20 px-3 text-xs font-bold text-ink/50 tabular">
        {props.position}
      </div>
      <ToolButton label="Add" shortcut="A" onClick={props.onAdd} active={props.drawing} />
      <ToolButton label="Delete" shortcut="D" onClick={props.onDeleteSelected} disabled={!props.hasSelection} />
      <div className="flex-1" />
      <ToolButton label="−" shortcut="Zoom" onClick={props.onZoomOut} />
      <ToolButton label="+" onClick={props.onZoomIn} />
      <ToolButton label="Fit" shortcut="F" onClick={props.onFit} />
      <ToolButton label="?" onClick={props.onToggleShortcuts} />
      <div className="flex-1" />
      <ToolButton label="Save" shortcut="S" onClick={props.onSave} />
      <ToolButton label="Reject" onClick={props.onReject} />
      <button
        onClick={props.onApprove}
        className="flex h-full items-center gap-1.5 bg-ink px-4 text-xs font-bold uppercase tracking-widest text-paper transition-colors duration-150 hover:bg-accent"
      >
        Approve <span className="text-[9px] font-normal text-paper/50">Space</span>
      </button>
    </div>
  );
}
