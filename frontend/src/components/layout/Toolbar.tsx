import { useUiStore } from "@/stores/uiStore";
import { ModelSelector } from "@/components/models/ModelSelector";
import { ConnectionIndicator } from "@/components/connection/ConnectionIndicator";
import { PanelLeftOpen } from "lucide-react";
import { Button } from "@/components/ui/button";

export function Toolbar() {
  const sidebarCollapsed = useUiStore((s) => s.sidebarCollapsed);
  const toggleSidebar = useUiStore((s) => s.toggleSidebar);

  return (
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
      </div>
    </header>
  );
}
