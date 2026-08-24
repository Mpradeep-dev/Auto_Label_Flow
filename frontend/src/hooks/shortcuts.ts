/** The single declarative list of keyboard shortcuts (PLAN "Shortcuts live
 * in one declarative map... consumed both by the handler hook and by the
 * on-screen help panel, so they cannot drift apart"). `useKeyboardShortcuts`
 * dispatches by `id`; `ShortcutHelp` renders this same array — add a
 * shortcut here once and both stay in sync. */
export interface ShortcutDef {
  id: string;
  key: string;
  label: string;
}

export const SHORTCUTS: ShortcutDef[] = [
  { id: "add", key: "A", label: "Add annotation" },
  { id: "delete", key: "D", label: "Delete selected annotation" },
  { id: "edit", key: "E", label: "Edit selected (focus coords)" },
  { id: "prev", key: "←", label: "Previous image" },
  { id: "next", key: "→", label: "Next image" },
  { id: "approve", key: "Space", label: "Approve" },
  { id: "save", key: "S", label: "Save" },
  { id: "zoom", key: "Z", label: "Zoom in" },
  { id: "fit", key: "F", label: "Fit image" },
  { id: "class1", key: "1", label: "Class 1" },
  { id: "class2", key: "2", label: "Class 2" },
  { id: "class3", key: "3", label: "Class 3" },
];
