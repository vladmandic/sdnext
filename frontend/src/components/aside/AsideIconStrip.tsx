import { PanelLeftClose, PanelLeftOpen } from "lucide-react";
import { ASIDE_TABS } from "@/lib/constants";
import { useUiStore } from "@/stores/uiStore";
import { useJobQueueStore, selectHasActiveJobs } from "@/stores/jobStore";
import { cn } from "@/lib/utils";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

export function AsideIconStrip() {
  const activeTab = useUiStore((s) => s.activeAsideTab);
  const collapsed = useUiStore((s) => s.rightPanelCollapsed);
  const openAsideTab = useUiStore((s) => s.openAsideTab);
  const toggleRightPanel = useUiStore((s) => s.toggleRightPanel);

  const hasActiveJobs = useJobQueueStore(selectHasActiveJobs);

  function handleTabClick(tabId: typeof activeTab) {
    if (collapsed) {
      openAsideTab(tabId);
    } else if (activeTab === tabId) {
      toggleRightPanel();
    } else {
      openAsideTab(tabId);
    }
  }

  return (
    <div className="flex flex-col items-center w-12 shrink-0 border-l border-border bg-card py-1.5 gap-0.5">
      <Tooltip>
        <TooltipTrigger asChild>
          <button
            type="button"
            onClick={toggleRightPanel}
            className="flex items-center justify-center w-9 h-9 rounded-md text-muted-foreground hover:text-foreground hover:bg-muted/50 transition-colors"
          >
            {collapsed ? <PanelLeftClose className="h-4 w-4" /> : <PanelLeftOpen className="h-4 w-4" />}
          </button>
        </TooltipTrigger>
        <TooltipContent side="left">{collapsed ? "Expand panel" : "Collapse panel"}</TooltipContent>
      </Tooltip>

      <div className="w-6 h-px bg-border my-1" />

      {ASIDE_TABS.map((tab) => (
        <div key={tab.id} className="flex flex-col items-center">
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                type="button"
                onClick={() => handleTabClick(tab.id)}
                className={cn(
                  "relative flex items-center justify-center w-9 h-9 rounded-md transition-colors",
                  activeTab === tab.id
                    ? "bg-accent text-accent-foreground"
                    : "text-muted-foreground hover:text-foreground hover:bg-muted/50",
                )}
              >
                <tab.icon className="h-4 w-4" />
                {tab.id === "queue" && hasActiveJobs && (
                  <span className="absolute top-1 right-1 h-2 w-2 rounded-full bg-primary animate-pulse" />
                )}
              </button>
            </TooltipTrigger>
            <TooltipContent side="left">{tab.label}</TooltipContent>
          </Tooltip>
          {tab.hasSeparatorAfter && <div className="w-6 h-px bg-border my-1" />}
        </div>
      ))}
    </div>
  );
}
