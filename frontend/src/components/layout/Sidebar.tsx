import { NAV_ITEMS, IMAGES_SUB_TABS, EXTERNAL_LINKS } from "@/lib/constants";
import { useUiStore } from "@/stores/uiStore";
import type { SidebarView, ImagesSubTab } from "@/stores/uiStore";
import { cn } from "@/lib/utils";
import { PanelLeftClose, PanelLeftOpen } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { Separator } from "@/components/ui/separator";

export function Sidebar() {
  const collapsed = useUiStore((s) => s.sidebarCollapsed);
  const activeView = useUiStore((s) => s.activeSidebarView);
  const activeSubTab = useUiStore((s) => s.activeImagesSubTab);
  const viewCollapsed = useUiStore((s) => s.viewCollapsed);
  const leftPanelCollapsed = useUiStore((s) => s.leftPanelCollapsed);
  const setSidebarView = useUiStore((s) => s.setSidebarView);
  const setImagesSubTab = useUiStore((s) => s.setImagesSubTab);
  const toggleViewCollapsed = useUiStore((s) => s.toggleViewCollapsed);
  const toggleLeftPanel = useUiStore((s) => s.toggleLeftPanel);
  const toggleSidebar = useUiStore((s) => s.toggleSidebar);

  const hasSubTabs = activeView === "images" && !viewCollapsed;

  return (
    <div
      className={cn(
        "flex bg-sidebar border-r border-sidebar-border transition-[width] duration-200 flex-shrink-0",
        collapsed ? "w-0 overflow-hidden" : !hasSubTabs && "w-14",
      )}
    >
      {/* Column 1: Primary nav icons */}
      <nav className="flex flex-col w-14 flex-shrink-0">
        {/* Toggle button */}
        <Button
          variant="ghost"
          size="icon"
          onClick={toggleSidebar}
          className="h-10 w-full rounded-none text-sidebar-foreground/60 hover:text-sidebar-foreground hover:bg-sidebar-accent"
          title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          {collapsed ? <PanelLeftOpen size={18} /> : <PanelLeftClose size={18} />}
        </Button>

        {/* Primary nav items */}
        <div className="flex flex-col gap-1 px-1.5 py-2 flex-1">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const isActive = activeView === item.id;
            return (
              <Tooltip key={item.id}>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => {
                      if (isActive) {
                        toggleViewCollapsed();
                      } else {
                        setSidebarView(item.id as SidebarView);
                        if (viewCollapsed) toggleViewCollapsed();
                      }
                    }}
                    className={cn(
                      "w-full aspect-square text-sidebar-foreground/60 hover:text-sidebar-foreground hover:bg-sidebar-accent",
                      isActive && "bg-sidebar-accent text-primary",
                    )}
                  >
                    <Icon size={20} />
                  </Button>
                </TooltipTrigger>
                <TooltipContent side="right">{item.label}</TooltipContent>
              </Tooltip>
            );
          })}
        </div>

        {/* External links */}
        <div className="flex flex-col gap-0.5 px-1.5 pb-2">
          <Separator className="mb-1.5 bg-sidebar-border" />
          {EXTERNAL_LINKS.map((link) => {
            const Icon = link.icon;
            return (
              <Tooltip key={link.label}>
                <TooltipTrigger asChild>
                  <a
                    href={link.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className={cn(
                      "flex items-center justify-center w-full aspect-square rounded-md transition-colors",
                      "text-sidebar-foreground/30 hover:text-sidebar-foreground/60 hover:bg-sidebar-accent",
                    )}
                  >
                    <Icon size={16} />
                  </a>
                </TooltipTrigger>
                <TooltipContent side="right">{link.label}</TooltipContent>
              </Tooltip>
            );
          })}
        </div>
      </nav>

      {/* Column 2: Sub-tab labels (only for views with sub-tabs) */}
      {hasSubTabs && (
        <div className="flex flex-col border-l border-sidebar-border py-2 overflow-y-auto">
          {IMAGES_SUB_TABS.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeSubTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => {
                  if (isActive && !leftPanelCollapsed && !viewCollapsed) {
                    toggleLeftPanel();
                  } else {
                    setImagesSubTab(tab.id as ImagesSubTab);
                    if (leftPanelCollapsed) toggleLeftPanel();
                    if (viewCollapsed) toggleViewCollapsed();
                  }
                }}
                className={cn(
                  "flex items-center gap-2 px-3 py-1.5 text-xs transition-colors text-left whitespace-nowrap",
                  "text-sidebar-foreground/60 hover:text-sidebar-foreground hover:bg-sidebar-accent",
                  isActive && "bg-sidebar-accent text-primary font-medium",
                )}
              >
                <Icon size={14} className="flex-shrink-0" />
                <span className="relative">
                  {tab.label}
                  {/* Invisible bold copy reserves width so the column doesn't shift on selection */}
                  <span className="font-medium invisible block h-0 overflow-hidden" aria-hidden>{tab.label}</span>
                </span>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
