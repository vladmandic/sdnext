import { lazy, Suspense } from "react";
import { ASIDE_TABS } from "@/lib/constants";
import { useUiStore } from "@/stores/uiStore";
import { ScrollArea } from "@/components/ui/scroll-area";

const QuickSettingsTab = lazy(() => import("./tabs/QuickSettingsTab").then((m) => ({ default: m.QuickSettingsTab })));
const NetworksTab = lazy(() => import("./tabs/NetworksTab").then((m) => ({ default: m.NetworksTab })));
const ModelsTab = lazy(() => import("./tabs/ModelsTab").then((m) => ({ default: m.ModelsTab })));
const ExtensionsTab = lazy(() => import("./tabs/ExtensionsTab").then((m) => ({ default: m.ExtensionsTab })));
const SettingsTab = lazy(() => import("./tabs/SettingsTab").then((m) => ({ default: m.SettingsTab })));
const SystemTab = lazy(() => import("./tabs/SystemTab").then((m) => ({ default: m.SystemTab })));
const HistoryTab = lazy(() => import("./tabs/HistoryTab").then((m) => ({ default: m.HistoryTab })));
const InfoTab = lazy(() => import("./tabs/InfoTab").then((m) => ({ default: m.InfoTab })));
const ConsoleTab = lazy(() => import("./tabs/ConsoleTab").then((m) => ({ default: m.ConsoleTab })));

const TAB_COMPONENTS: Record<string, React.LazyExoticComponent<React.ComponentType>> = {
  "quick-settings": QuickSettingsTab,
  "networks": NetworksTab,
  "models": ModelsTab,
  "extensions": ExtensionsTab,
  "settings": SettingsTab,
  "system": SystemTab,
  "history": HistoryTab,
  "info": InfoTab,
  "console": ConsoleTab,
};

export function AsidePanel() {
  const activeTab = useUiStore((s) => s.activeAsideTab);
  const tabMeta = ASIDE_TABS.find((t) => t.id === activeTab);
  const TabComponent = TAB_COMPONENTS[activeTab];

  return (
    <div className="flex flex-col h-full bg-card">
      <div className="flex items-center gap-2 px-3 py-2 border-b border-border shrink-0">
        {tabMeta && <tabMeta.icon className="h-4 w-4 text-muted-foreground" />}
        <span className="text-sm font-medium">{tabMeta?.label ?? activeTab}</span>
      </div>
      <ScrollArea className="flex-1">
        <Suspense fallback={<div className="p-3 text-xs text-muted-foreground">Loading...</div>}>
          {TabComponent && <TabComponent />}
        </Suspense>
      </ScrollArea>
    </div>
  );
}
