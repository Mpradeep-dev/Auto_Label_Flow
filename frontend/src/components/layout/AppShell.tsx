import { useState } from "react";
import { Outlet } from "react-router-dom";
import { Sidebar, useSidebarCollapsed } from "./Sidebar";
import { Breadcrumbs } from "./Breadcrumbs";
import { CommandPalette } from "./CommandPalette";

export function AppShell() {
  const [collapsed, toggle] = useSidebarCollapsed();
  const [searchOpen, setSearchOpen] = useState(false);

  return (
    <div className="flex h-[100dvh] w-full overflow-hidden bg-paper text-ink">
      <Sidebar collapsed={collapsed} onToggle={toggle} onOpenSearch={() => setSearchOpen(true)} />
      <main className="flex min-w-0 flex-1 flex-col overflow-hidden">
        <Breadcrumbs />
        <div className="min-h-0 flex-1 overflow-y-auto">
          <Outlet />
        </div>
      </main>
      <CommandPalette open={searchOpen} onOpenChange={setSearchOpen} />
    </div>
  );
}
