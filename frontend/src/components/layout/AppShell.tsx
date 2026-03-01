import { Sidebar } from "./Sidebar";
import { Toolbar } from "./Toolbar";
import { StatusBar } from "./StatusBar";
import { MainContent } from "./MainContent";
import { LeftPanel } from "./LeftPanel";
import { AsideIconStrip } from "@/components/aside/AsideIconStrip";
import { AsidePanel } from "@/components/aside/AsidePanel";
import { ResizablePanelGroup, ResizablePanel, ResizableHandle } from "@/components/ui/resizable";
import { useUiStore } from "@/stores/uiStore";
import { useHistoryInit } from "@/hooks/useHistoryInit";
import { useJobTracker } from "@/hooks/useJobTracker";
import { useGlobalWs } from "@/hooks/useGlobalWs";
import { useShortcutDispatcher } from "@/hooks/useShortcutDispatcher";
import { useShortcut } from "@/hooks/useShortcut";
import { useModelDefaultsSuggester } from "@/hooks/useModelDefaultsSuggester";
import { ShortcutOverlay } from "@/components/ShortcutOverlay";
import { CommandPalette } from "@/components/CommandPalette";
import { ComparisonDialog } from "@/components/comparison/ComparisonDialog";
import { cn } from "@/lib/utils";

export function AppShell() {
  useHistoryInit();
  useJobTracker();
  useGlobalWs();
  useShortcutDispatcher();
  useModelDefaultsSuggester();

  // Global layout shortcuts
  useShortcut("toggle-sidebar", () => useUiStore.getState().toggleSidebar());
  useShortcut("toggle-left-panel", () => useUiStore.getState().toggleLeftPanel());
  useShortcut("toggle-right-panel", () => useUiStore.getState().toggleRightPanel());

  const leftPanelCollapsed = useUiStore((s) => s.leftPanelCollapsed);
  const viewCollapsed = useUiStore((s) => s.viewCollapsed);
  const leftPanelWidth = useUiStore((s) => s.leftPanelWidth);
  const leftHidden = leftPanelCollapsed || viewCollapsed;
  const rightPanelCollapsed = useUiStore((s) => s.rightPanelCollapsed);

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-background text-foreground">
      <Sidebar />

      <div className="flex flex-1 flex-col min-w-0">
        <Toolbar />

        <div className="flex flex-1 min-h-0">
          {/* Left panel */}
          <aside
            className={cn(
              "border-r border-border bg-card flex-shrink-0 overflow-hidden transition-[width] duration-200",
              leftHidden && "w-0 border-r-0",
            )}
            style={{ width: leftHidden ? 0 : leftPanelWidth }}
          >
            {!leftHidden && <LeftPanel />}
          </aside>

          {/* Main + Aside content */}
          <ResizablePanelGroup orientation="horizontal" id="layout-main" className="flex-1">
            <ResizablePanel id="panel-main" minSize="30%">
              <main className="h-full overflow-auto">
                <MainContent />
              </main>
            </ResizablePanel>
            {!rightPanelCollapsed && (
              <>
                <ResizableHandle />
                <ResizablePanel
                  id="panel-aside"
                  minSize={280}
                  maxSize="70%"
                  defaultSize="30%"
                >
                  <AsidePanel />
                </ResizablePanel>
              </>
            )}
          </ResizablePanelGroup>

          {/* Icon strip */}
          <AsideIconStrip />
        </div>

        <StatusBar />
      </div>

      <ShortcutOverlay />
      <CommandPalette />
      <ComparisonDialog />
    </div>
  );
}
