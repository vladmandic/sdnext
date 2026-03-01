import type { LucideIcon } from "lucide-react";
import {
  Play, Square, SkipForward, RotateCcw, Trash2,
  RefreshCw, Download, Upload,
  PanelLeft, PanelRight, Sidebar,
  ImageIcon, Video, Sparkles, MessageSquare, Images,
  GitCompareArrows,
} from "lucide-react";
import { api } from "@/api/client";
import { useUiStore } from "@/stores/uiStore";
import { useGenerationStore } from "@/stores/generationStore";
import { useComparisonStore } from "@/stores/comparisonStore";
import { resolveImageSrc } from "@/lib/utils";
import { useJobQueueStore, selectRunningJob } from "@/stores/jobStore";
import { sendToJob } from "@/hooks/useJobTracker";
import { NAV_ITEMS, IMAGES_SUB_TABS, ASIDE_TABS } from "@/lib/constants";
import type { SidebarView, ImagesSubTab } from "@/stores/uiStore";
import type { AsideTab } from "@/lib/constants";

export interface PaletteAction {
  id: string;
  label: string;
  icon: LucideIcon;
  group: "Recent" | "Actions" | "Navigation";
  keywords: string[];
  shortcutId?: string;
  action: () => void;
}

const NAV_ICONS: Record<string, LucideIcon> = {
  images: ImageIcon,
  video: Video,
  process: Sparkles,
  caption: MessageSquare,
  gallery: Images,
};

export function buildActions(): PaletteAction[] {
  const actions: PaletteAction[] = [];

  // --- Generation actions ---
  actions.push({
    id: "generate",
    label: "Generate",
    icon: Play,
    group: "Actions",
    keywords: ["run", "create", "start"],
    shortcutId: "generate",
    action: () => {
      // Trigger a click on the generate button (the actual submission logic lives in ActionBar)
      const btn = document.querySelector<HTMLButtonElement>("[data-action='generate']");
      btn?.click();
    },
  });

  actions.push({
    id: "interrupt",
    label: "Interrupt generation",
    icon: Square,
    group: "Actions",
    keywords: ["stop", "cancel", "abort"],
    action: () => {
      const job = selectRunningJob(useJobQueueStore.getState());
      if (job && job.domain === "generate") {
        sendToJob(job.id, { type: "interrupt" });
      }
    },
  });

  actions.push({
    id: "skip",
    label: "Skip current step",
    icon: SkipForward,
    group: "Actions",
    keywords: ["next", "advance"],
    shortcutId: "skip",
    action: () => {
      const job = selectRunningJob(useJobQueueStore.getState());
      if (job && job.domain === "generate") {
        sendToJob(job.id, { type: "skip" });
      }
    },
  });

  actions.push({
    id: "reset-params",
    label: "Reset all parameters",
    icon: Trash2,
    group: "Actions",
    keywords: ["clear", "default"],
    action: () => useGenerationStore.getState().reset(),
  });

  actions.push({
    id: "restore-last",
    label: "Restore last settings",
    icon: RotateCcw,
    group: "Actions",
    keywords: ["undo", "previous", "history"],
    action: () => {
      const btn = document.querySelector<HTMLButtonElement>("[data-action='restore']");
      btn?.click();
    },
  });

  // --- Model actions ---
  actions.push({
    id: "refresh-models",
    label: "Refresh model list",
    icon: RefreshCw,
    group: "Actions",
    keywords: ["model", "checkpoint", "scan"],
    action: () => { api.post("/sdapi/v2/checkpoint/refresh"); },
  });

  actions.push({
    id: "reload-model",
    label: "Reload current model",
    icon: Download,
    group: "Actions",
    keywords: ["model", "checkpoint", "load"],
    action: () => { api.post("/sdapi/v2/checkpoint/reload"); },
  });

  actions.push({
    id: "unload-model",
    label: "Unload model",
    icon: Upload,
    group: "Actions",
    keywords: ["model", "checkpoint", "free", "memory"],
    action: () => { api.post("/sdapi/v2/checkpoint/unload"); },
  });

  // --- Comparison ---
  actions.push({
    id: "compare-results",
    label: "Compare Results",
    icon: GitCompareArrows,
    group: "Actions",
    keywords: ["compare", "diff", "side by side", "before", "after"],
    action: () => {
      const gen = useGenerationStore.getState();
      if (gen.results.length >= 2) {
        const a = gen.results[0];
        const b = gen.results[1];
        const srcA = resolveImageSrc(a.images[0]);
        const srcB = resolveImageSrc(b.images[0]);
        useComparisonStore.getState().openComparison(
          { src: srcA, label: "Latest", resultId: a.id, imageIndex: 0 },
          { src: srcB, label: "Previous", resultId: b.id, imageIndex: 0 },
        );
      }
    },
  });

  // --- Layout actions ---
  actions.push({
    id: "toggle-sidebar",
    label: "Toggle sidebar",
    icon: Sidebar,
    group: "Actions",
    keywords: ["sidebar", "nav", "collapse"],
    shortcutId: "toggle-sidebar",
    action: () => useUiStore.getState().toggleSidebar(),
  });

  actions.push({
    id: "toggle-left-panel",
    label: "Toggle left panel",
    icon: PanelLeft,
    group: "Actions",
    keywords: ["panel", "params", "settings"],
    shortcutId: "toggle-left-panel",
    action: () => useUiStore.getState().toggleLeftPanel(),
  });

  actions.push({
    id: "toggle-right-panel",
    label: "Toggle right panel",
    icon: PanelRight,
    group: "Actions",
    keywords: ["panel", "aside", "networks"],
    shortcutId: "toggle-right-panel",
    action: () => useUiStore.getState().toggleRightPanel(),
  });

  // --- Navigation: views ---
  for (const nav of NAV_ITEMS) {
    actions.push({
      id: `nav-${nav.id}`,
      label: `Go to ${nav.label}`,
      icon: NAV_ICONS[nav.id] ?? nav.icon,
      group: "Navigation",
      keywords: ["view", "page", "navigate", nav.label.toLowerCase()],
      action: () => useUiStore.getState().setSidebarView(nav.id as SidebarView),
    });
  }

  // --- Navigation: images sub-tabs ---
  for (const tab of IMAGES_SUB_TABS) {
    actions.push({
      id: `subtab-${tab.id}`,
      label: `Images \u203a ${tab.label}`,
      icon: tab.icon,
      group: "Navigation",
      keywords: ["tab", "images", tab.label.toLowerCase()],
      action: () => {
        useUiStore.getState().setSidebarView("images");
        useUiStore.getState().setImagesSubTab(tab.id as ImagesSubTab);
      },
    });
  }

  // --- Navigation: aside tabs ---
  for (const tab of ASIDE_TABS) {
    actions.push({
      id: `aside-${tab.id}`,
      label: `Open ${tab.label} panel`,
      icon: tab.icon,
      group: "Navigation",
      keywords: ["panel", "aside", tab.label.toLowerCase()],
      action: () => useUiStore.getState().openAsideTab(tab.id as AsideTab),
    });
  }

  return actions;
}
