import type { LucideIcon } from "lucide-react";
import {
  Play, Square, SkipForward, RotateCcw,
  RefreshCw, Download, Upload,
  ImageIcon, Video, Sparkles, MessageSquare, Images,
  Settings,
} from "lucide-react";
import { NAV_ITEMS, IMAGES_SUB_TABS, ASIDE_TABS } from "@/lib/constants";
import { PARAM_MAP } from "@/lib/paramMap";
import type { NavigateTarget } from "@/lib/navigateToParam";
import type { ImagesSubTab } from "@/stores/uiStore";
import type { AsideTab } from "@/lib/constants";

export interface PaletteAction {
  id: string;
  label: string;
  icon: LucideIcon;
  group: string;
  keywords: string[];
  shortcutId?: string;
  target: NavigateTarget;
  showOnlyInSearch?: boolean;
}

const NAV_ICONS: Record<string, LucideIcon> = {
  images: ImageIcon,
  video: Video,
  process: Sparkles,
  caption: MessageSquare,
  gallery: Images,
};

const TAB_ICONS: Record<string, LucideIcon> = {};
for (const t of IMAGES_SUB_TABS) TAB_ICONS[t.id] = t.icon;

const TAB_LABELS: Record<string, string> = {};
for (const t of IMAGES_SUB_TABS) TAB_LABELS[t.id] = t.label;

export function buildActions(): PaletteAction[] {
  const actions: PaletteAction[] = [];

  // --- Generation buttons ---
  actions.push({
    id: "generate",
    label: "Generate",
    icon: Play,
    group: "Actions",
    keywords: ["run", "create", "start"],
    shortcutId: "generate",
    target: { param: "generate" },
  });

  actions.push({
    id: "interrupt",
    label: "Interrupt generation",
    icon: Square,
    group: "Actions",
    keywords: ["stop", "cancel", "abort"],
    target: { param: "stop" },
  });

  actions.push({
    id: "skip",
    label: "Skip current step",
    icon: SkipForward,
    group: "Actions",
    keywords: ["next", "advance"],
    shortcutId: "skip",
    target: { param: "skip" },
  });

  actions.push({
    id: "restore-last",
    label: "Restore last settings",
    icon: RotateCcw,
    group: "Actions",
    keywords: ["undo", "previous", "history"],
    target: { param: "restore" },
  });

  // --- Model actions → navigate to Models panel ---
  actions.push({
    id: "refresh-models",
    label: "Refresh model list",
    icon: RefreshCw,
    group: "Actions",
    keywords: ["model", "checkpoint", "scan"],
    target: { aside: "models" },
  });

  actions.push({
    id: "reload-model",
    label: "Reload current model",
    icon: Download,
    group: "Actions",
    keywords: ["model", "checkpoint", "load"],
    target: { aside: "models" },
  });

  actions.push({
    id: "unload-model",
    label: "Unload model",
    icon: Upload,
    group: "Actions",
    keywords: ["model", "checkpoint", "free", "memory"],
    target: { aside: "models" },
  });

  // --- Settings search ---
  actions.push({
    id: "search-settings",
    label: "Search settings...",
    icon: Settings,
    group: "Navigation",
    keywords: ["settings", "search", "find", "option", "preference", "configure"],
    target: { aside: "settings" },
  });

  // --- Navigation: views ---
  for (const nav of NAV_ITEMS) {
    actions.push({
      id: `nav-${nav.id}`,
      label: `Go to ${nav.label}`,
      icon: NAV_ICONS[nav.id] ?? nav.icon,
      group: "Navigation",
      keywords: ["view", "page", "navigate", nav.label.toLowerCase()],
      target: { view: nav.id },
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
      target: { tab: tab.id as ImagesSubTab },
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
      target: { aside: tab.id as AsideTab },
    });
  }

  // --- Parameter navigation (search-only) ---
  for (const entry of PARAM_MAP) {
    const tabLabel = TAB_LABELS[entry.tab] ?? entry.tab;
    const icon = TAB_ICONS[entry.tab] ?? Settings;
    actions.push({
      id: `param-${entry.tab}-${entry.param}`,
      label: entry.label,
      icon,
      group: tabLabel,
      keywords: [...entry.keywords, entry.param, entry.section],
      target: { tab: entry.tab, section: entry.section, param: entry.param },
      showOnlyInSearch: true,
    });
  }

  return actions;
}
