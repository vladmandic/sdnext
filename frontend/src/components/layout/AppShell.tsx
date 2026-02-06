import { Sidebar } from "./Sidebar";
import { Toolbar } from "./Toolbar";
import { StatusBar } from "./StatusBar";
import { MainContent } from "./MainContent";
import { LeftPanel } from "./LeftPanel";
import { RightPanel } from "./RightPanel";
import { useUiStore } from "@/stores/uiStore";
import { cn } from "@/lib/utils";

export function AppShell() {
  const leftPanelCollapsed = useUiStore((s) => s.leftPanelCollapsed);
  const leftPanelWidth = useUiStore((s) => s.leftPanelWidth);
  const rightPanelCollapsed = useUiStore((s) => s.rightPanelCollapsed);
  const rightPanelWidth = useUiStore((s) => s.rightPanelWidth);

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-background text-foreground">
      {/* Sidebar */}
      <Sidebar />

      {/* Main area */}
      <div className="flex flex-1 flex-col min-w-0">
        {/* Toolbar */}
        <Toolbar />

        {/* Left Panel + Content + Right Panel */}
        <div className="flex flex-1 min-h-0">
          {/* Left panel (parameters, gallery) */}
          <aside
            className={cn(
              "border-r border-border bg-card flex-shrink-0 overflow-y-auto transition-[width] duration-200",
              leftPanelCollapsed && "w-0 border-r-0",
            )}
            style={{ width: leftPanelCollapsed ? 0 : leftPanelWidth }}
          >
            {!leftPanelCollapsed && <LeftPanel />}
          </aside>

          {/* Main content */}
          <main className="flex-1 min-w-0 overflow-auto">
            <MainContent />
          </main>

          {/* Right panel (info, metadata) */}
          <aside
            className={cn(
              "border-l border-border bg-card flex-shrink-0 overflow-y-auto transition-[width] duration-200",
              rightPanelCollapsed && "w-0 border-l-0",
            )}
            style={{ width: rightPanelCollapsed ? 0 : rightPanelWidth }}
          >
            {!rightPanelCollapsed && <RightPanel />}
          </aside>
        </div>

        {/* Status bar */}
        <StatusBar />
      </div>
    </div>
  );
}
