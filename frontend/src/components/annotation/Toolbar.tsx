export type ActiveTool = "select" | "bbox" | "polygon";

interface Props {
  onPrev: () => void;
  onNext: () => void;
  onApprove: () => void;
  onReject: () => void;
  onSave: () => void;
  onDeleteSelected: () => void;
  onSelectBboxTool: () => void;
  onSelectPolygonTool: () => void;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onFit: () => void;
  onToggleShortcuts: () => void;
  hasSelection: boolean;
  activeTool: ActiveTool;
  position: string; // "N / total"
  reviewStatus: "PENDING" | "APPROVED" | "REJECTED";
  approving: boolean;
  rejecting: boolean;
  approveError: string | null;
  rejectError: string | null;
  justSaved: boolean;
}

const STATUS_STYLE: Record<Props["reviewStatus"], string> = {
  PENDING: "text-ink/40",
  APPROVED: "text-ink",
  REJECTED: "text-accent",
};

function ToolButton({
  label,
  shortcut,
  onClick,
  disabled,
  active,
  title,
}: {
  label: string;
  shortcut?: string;
  onClick: () => void;
  disabled?: boolean;
  active?: boolean;
  title?: string;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      title={title}
      className={`flex h-full items-center gap-1.5 border-r border-ink/20 px-3 text-xs font-bold uppercase tracking-widest transition-colors duration-150 ${
        active ? "bg-ink text-paper" : "hover:bg-orange"
      } disabled:cursor-not-allowed disabled:opacity-30`}
    >
      {label}
      {shortcut && <span className="text-[9px] font-normal text-ink/40">{shortcut}</span>}
    </button>
  );
}

export function Toolbar(props: Props) {
  const error = props.approveError ?? props.rejectError;
  return (
    <div className="flex h-11 shrink-0 items-stretch border-t-4 border-ink bg-paper">
      <ToolButton label="← Prev" shortcut="←" onClick={props.onPrev} />
      <ToolButton label="Next →" shortcut="→" onClick={props.onNext} />
      <div className="flex items-center border-r border-ink/20 px-3 text-xs font-bold text-ink/50 tabular">
        {props.position}
      </div>
      <ToolButton label="Box" shortcut="B" onClick={props.onSelectBboxTool} active={props.activeTool === "bbox"} />
      <ToolButton
        label="Polygon"
        shortcut="P"
        onClick={props.onSelectPolygonTool}
        active={props.activeTool === "polygon"}
      />
      <ToolButton label="Delete" shortcut="D" onClick={props.onDeleteSelected} disabled={!props.hasSelection} />
      <div className="flex-1" />
      {error ? (
        <div className="flex items-center border-r border-ink/20 px-3 text-xs font-bold uppercase tracking-widest text-accent">
          {error}
        </div>
      ) : (
        // Without this, a click on Approve/Reject that actually succeeds
        // (or fails) is invisible on this page — nothing here changes, so
        // "the button doesn't work" is the only reasonable read even when
        // the request went through fine.
        <div className={`flex items-center border-r border-ink/20 px-3 text-xs font-bold uppercase tracking-widest ${STATUS_STYLE[props.reviewStatus]}`}>
          {props.reviewStatus}
        </div>
      )}
      <ToolButton label="−" shortcut="⇧Z" onClick={props.onZoomOut} />
      <ToolButton label="+" shortcut="Z" onClick={props.onZoomIn} />
      <ToolButton label="Fit" shortcut="F" onClick={props.onFit} />
      <ToolButton label="?" onClick={props.onToggleShortcuts} />
      <div className="flex-1" />
      <ToolButton label={props.justSaved ? "Saved ✓" : "Save"} shortcut="S" onClick={props.onSave} />
      <ToolButton label={props.rejecting ? "Rejecting…" : "Reject"} onClick={props.onReject} disabled={props.rejecting} />
      <button
        onClick={props.onApprove}
        disabled={props.approving}
        className="flex h-full items-center gap-1.5 bg-ink px-4 text-xs font-bold uppercase tracking-widest text-paper transition-colors duration-150 hover:bg-accent disabled:cursor-not-allowed disabled:opacity-60"
      >
        {props.approving ? "Approving…" : "Approve"} <span className="text-[9px] font-normal text-paper/50">Space</span>
      </button>
    </div>
  );
}
