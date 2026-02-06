import { useState } from "react";
import { useUiStore } from "@/stores/uiStore";
import { ModelSelector } from "@/components/models/ModelSelector";
import { ConnectionIndicator } from "@/components/connection/ConnectionIndicator";
import { SettingsView } from "@/components/settings/SettingsView";
import { PanelLeftClose, PanelLeftOpen, Settings } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetHeader, SheetTitle } from "@/components/ui/sheet";

export function Toolbar() {
  const sidebarCollapsed = useUiStore((s) => s.sidebarCollapsed);
  const toggleSidebar = useUiStore((s) => s.toggleSidebar);
  const leftPanelCollapsed = useUiStore((s) => s.leftPanelCollapsed);
  const toggleLeftPanel = useUiStore((s) => s.toggleLeftPanel);
  const [settingsOpen, setSettingsOpen] = useState(false);

  return (
    <>
      <header className="flex items-center h-11 px-2 gap-2 border-b border-border bg-card flex-shrink-0">
        {/* Sidebar toggle (visible when sidebar is collapsed) */}
        {sidebarCollapsed && (
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={toggleSidebar}
            className="text-muted-foreground"
            title="Show sidebar"
          >
            <PanelLeftOpen size={16} />
          </Button>
        )}

        {/* Model selector */}
        <div className="flex-1 min-w-0 flex items-center gap-2">
          <ModelSelector />
        </div>

        {/* Right side controls */}
        <div className="flex items-center gap-1">
          <ConnectionIndicator />

          <Button
            variant="ghost"
            size="icon-sm"
            onClick={() => setSettingsOpen(true)}
            className="text-muted-foreground"
            title="Settings"
          >
            <Settings size={16} />
          </Button>

          <Button
            variant="ghost"
            size="icon-sm"
            onClick={toggleLeftPanel}
            className="text-muted-foreground"
            title={leftPanelCollapsed ? "Show parameters" : "Hide parameters"}
          >
            {leftPanelCollapsed ? <PanelLeftOpen size={16} /> : <PanelLeftClose size={16} />}
          </Button>
        </div>
      </header>

      {/* Settings sheet */}
      <Sheet open={settingsOpen} onOpenChange={setSettingsOpen}>
        <SheetContent side="right" className="w-[600px] sm:max-w-[600px] p-0">
          <SheetHeader className="sr-only">
            <SheetTitle>Settings</SheetTitle>
          </SheetHeader>
          <SettingsView />
        </SheetContent>
      </Sheet>
    </>
  );
}
