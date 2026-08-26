import { Link, NavLink, useParams } from "react-router-dom";
import { useEffect, useState } from "react";
import logoMark from "@/assets/logo-mark.png";

// Below this, the sidebar's expanded 224px eats more than half a phone-width
// viewport, squeezing the actual page content (the annotation canvas, in
// the worst case) into an unusable sliver (audit finding FE-05). Treated as
// a hard floor, not a soft preference — same precedent as AnnotatePage
// forcing its right panel collapsed below 1024px.
const NARROW_BREAKPOINT_PX = 640;

function isNarrowViewport(): boolean {
  return typeof window !== "undefined" && window.innerWidth < NARROW_BREAKPOINT_PX;
}

// Every sidebar entry the spec calls for, grouped so related items read as
// related instead of one flat undifferentiated list. Entries that need a
// project context are disabled until one is selected; entries not yet built
// land on a placeholder page rather than a dead link, so navigation is
// always coherent even mid-build (see PLAN build order).
const NAV_GROUPS = [
  {
    label: "Workflow",
    items: [
      { label: "Pipeline", path: "" },
      { label: "Dataset", path: "/datasets" },
      { label: "Images", path: "/images" },
      { label: "Videos", path: "/videos" },
    ],
  },
  {
    label: "AI",
    items: [
      { label: "Auto Annotation", path: "/auto-annotation" },
      { label: "Review Queue", path: "/review" },
      { label: "Models", path: "/models" },
      { label: "Training Runs", path: "/training" },
    ],
  },
  {
    label: "Output",
    items: [
      { label: "Export", path: "/export" },
      // Distinct from the always-visible top-level "Settings" link below
      // (Kaggle/Roboflow, account-wide) — this one is this project's own
      // name, class taxonomy, and quality rule packs. `short` avoids
      // colliding with "Pipeline" in the collapsed one-letter rail.
      { label: "Project Settings", short: "PS", path: "/settings" },
    ],
  },
] as const;

export function Sidebar({
  collapsed,
  onToggle,
  onOpenSearch,
}: {
  collapsed: boolean;
  onToggle: () => void;
  onOpenSearch: () => void;
}) {
  const { projectId } = useParams<{ projectId?: string }>();
  // Groups default open so nothing that was always-visible before disappears
  // on its own — collapsing is something the user does, not a default state.
  const [openGroups, setOpenGroups] = useState<Record<string, boolean>>(() =>
    Object.fromEntries(NAV_GROUPS.map((g) => [g.label, true])),
  );
  const toggleGroup = (label: string) => setOpenGroups((g) => ({ ...g, [label]: !g[label] }));

  return (
    <nav
      className={`flex h-full flex-col border-r-4 border-ink bg-paper transition-[width] duration-150 ease-out ${
        collapsed ? "w-14" : "w-56"
      }`}
      aria-label="Primary"
    >
      {/* Once you're past the landing page there was no way back to it
          (audit finding: entering the app is a one-way door) — this doubles
          as the brand mark and a "go home" link, same convention as the
          logo click on the landing page header. */}
      <Link
        to="/"
        className="group flex h-14 shrink-0 items-center gap-2 border-b-4 border-ink px-4 hover:bg-orange"
        aria-label="Home"
      >
        <img src={logoMark} alt="" className="h-6 w-6 shrink-0" />
        {!collapsed && (
          <span className="truncate text-xs font-bold uppercase tracking-widest">
            Auto <span className="text-orange group-hover:text-ink">Label</span> Flow
          </span>
        )}
      </Link>

      <button
        onClick={onToggle}
        className="flex h-12 shrink-0 items-center justify-center border-b-4 border-ink text-xs font-bold uppercase tracking-widest hover:bg-ink hover:text-paper"
        aria-label={collapsed ? "Expand navigation" : "Collapse navigation"}
      >
        {collapsed ? "»" : "« Collapse"}
      </button>

      <NavLink
        to="/projects"
        className={({ isActive }) =>
          `flex h-12 shrink-0 items-center border-b-2 border-ink px-4 text-xs font-bold uppercase tracking-widest ${
            isActive ? "bg-ink text-paper" : "hover:bg-orange"
          }`
        }
      >
        {collapsed ? "P" : "Projects"}
      </NavLink>

      {/* Account-wide (Kaggle/Roboflow connect) — deliberately NOT gated
          behind selecting a project, unlike everything in the list below. */}
      <NavLink
        to="/settings"
        className={({ isActive }) =>
          `flex h-12 shrink-0 items-center border-b-2 border-ink px-4 text-xs font-bold uppercase tracking-widest ${
            isActive ? "bg-ink text-paper" : "hover:bg-orange"
          }`
        }
      >
        {collapsed ? "S" : "Settings"}
      </NavLink>

      {/* Same reasoning as Settings above — a "what does any of this mean"
          question isn't scoped to a project either, so this can't be gated
          behind picking one. */}
      <NavLink
        to="/help"
        className={({ isActive }) =>
          `flex h-12 shrink-0 items-center border-b-4 border-ink px-4 text-xs font-bold uppercase tracking-widest ${
            isActive ? "bg-ink text-paper" : "hover:bg-orange"
          }`
        }
      >
        {collapsed ? "?" : "Help"}
      </NavLink>

      <button
        onClick={onOpenSearch}
        className="flex h-11 shrink-0 items-center justify-between border-b-2 border-ink/20 px-4 text-xs font-semibold uppercase tracking-widest hover:bg-orange"
      >
        {collapsed ? "⌕" : "Search"}
        {!collapsed && <span className="text-[9px] font-normal text-ink/40">⌘K</span>}
      </button>

      <ul className="flex-1 overflow-y-auto">
        {NAV_GROUPS.map((group) => {
          const open = collapsed || openGroups[group.label];
          return (
            <li key={group.label}>
              {!collapsed && (
                <button
                  type="button"
                  onClick={() => toggleGroup(group.label)}
                  aria-expanded={open}
                  className="flex w-full items-center justify-between border-b border-ink/10 bg-muted px-4 py-1.5 text-[9px] font-bold uppercase tracking-widest text-ink/40 hover:text-ink"
                >
                  {group.label}
                  <span className={`transition-transform duration-150 ${open ? "rotate-0" : "-rotate-90"}`}>▾</span>
                </button>
              )}
              {open && (
                <ul>
                  {group.items.map((item) => {
                    const to = projectId ? `/projects/${projectId}${item.path}` : "/projects";
                    const disabled = !projectId;
                    return (
                      <li key={item.label} className="border-b border-ink/20">
                        <NavLink
                          to={to}
                          end={item.path === ""}
                          aria-disabled={disabled}
                          className={({ isActive }) =>
                            `flex h-11 items-center px-4 text-xs font-semibold uppercase tracking-widest transition-colors duration-150 ${
                              disabled
                                ? "cursor-not-allowed text-ink/30"
                                : isActive
                                  ? "bg-ink text-paper"
                                  : "hover:bg-orange"
                            }`
                          }
                        >
                          {collapsed ? ("short" in item ? item.short : item.label.slice(0, 1)) : item.label}
                        </NavLink>
                      </li>
                    );
                  })}
                </ul>
              )}
            </li>
          );
        })}
      </ul>
    </nav>
  );
}

const SIDEBAR_COLLAPSED_KEY = "sidebar-collapsed";

export function useSidebarCollapsed(): [boolean, () => void] {
  const [manualCollapsed, setManualCollapsed] = useState(() => {
    try {
      return localStorage.getItem(SIDEBAR_COLLAPSED_KEY) === "true";
    } catch {
      return false;
    }
  });
  // Tracked separately from the persisted preference above and reactive to
  // resize (not just checked once at mount) — narrowing the window after
  // load used to leave the full-width sidebar in place with nothing
  // reflowing to compensate.
  const [narrow, setNarrow] = useState(isNarrowViewport);

  useEffect(() => {
    function onResize() {
      setNarrow(isNarrowViewport());
    }
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  const toggle = () => {
    setManualCollapsed((c) => {
      const next = !c;
      try {
        localStorage.setItem(SIDEBAR_COLLAPSED_KEY, String(next));
      } catch {
        /* private-browsing / storage disabled — collapse state just won't persist */
      }
      return next;
    });
  };

  return [manualCollapsed || narrow, toggle];
}
