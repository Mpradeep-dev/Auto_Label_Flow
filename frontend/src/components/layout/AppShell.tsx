import { Outlet } from "react-router-dom";
import { Sidebar, useSidebarCollapsed } from "./Sidebar";

export function AppShell() {
  const [collapsed, toggle] = useSidebarCollapsed();

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-paper text-ink">
      <Sidebar collapsed={collapsed} onToggle={toggle} />
      <main className="min-w-0 flex-1 overflow-y-auto">
        <Outlet />
      </main>
    </div>
  );
}
