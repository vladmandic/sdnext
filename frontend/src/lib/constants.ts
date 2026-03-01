import type { LucideIcon } from "lucide-react";
import {
  ImageIcon, Video, Sparkles, MessageSquare, Images,
  Type, SlidersHorizontal, Compass, Wand2, ScanSearch,
  Settings2, Layers, FileCode, Palette,
  BookOpen, Github, MessageCircle, Users,
  Gauge, LayoutGrid, Box, Puzzle, Settings, Monitor, Clock, Info, Terminal,
  ListOrdered,
} from "lucide-react";

export interface NavItem {
  id: string;
  label: string;
  icon: LucideIcon;
  capability?: keyof import("@/api/types/server").ServerCapabilities;
}

export interface SubTabItem {
  id: string;
  label: string;
  icon: LucideIcon;
}

export interface ExternalLink {
  label: string;
  icon: LucideIcon;
  url: string;
}

export type AsideTab = "quick-settings" | "networks" | "models" | "queue" | "extensions" | "settings" | "system" | "history" | "info" | "console";

export interface AsideTabItem {
  id: AsideTab;
  label: string;
  icon: LucideIcon;
  hasSeparatorAfter?: boolean;
}

export const ASIDE_TABS: AsideTabItem[] = [
  { id: "quick-settings", label: "Quick Settings", icon: Gauge },
  { id: "networks", label: "Networks", icon: LayoutGrid },
  { id: "models", label: "Models", icon: Box },
  { id: "queue", label: "Queue", icon: ListOrdered, hasSeparatorAfter: true },
  { id: "extensions", label: "Extensions", icon: Puzzle },
  { id: "settings", label: "Settings", icon: Settings },
  { id: "system", label: "System", icon: Monitor },
  { id: "history", label: "History", icon: Clock },
  { id: "info", label: "Info", icon: Info },
  { id: "console", label: "Console", icon: Terminal },
];

/** Primary sidebar navigation */
export const NAV_ITEMS: NavItem[] = [
  { id: "images", label: "Images", icon: ImageIcon },
  { id: "video", label: "Video", icon: Video, capability: "video" },
  { id: "process", label: "Process", icon: Sparkles },
  { id: "caption", label: "Caption", icon: MessageSquare },
  { id: "gallery", label: "Gallery", icon: Images },
];

/** Sub-tabs for the Images view (matches SD.Next control tab structure) */
export const IMAGES_SUB_TABS: SubTabItem[] = [
  { id: "prompts", label: "Prompts", icon: Type },
  { id: "sampler", label: "Sampler", icon: SlidersHorizontal },
  { id: "guidance", label: "Guidance", icon: Compass },
  { id: "refine", label: "Refine", icon: Wand2 },
  { id: "detail", label: "Detail", icon: ScanSearch },
  { id: "advanced", label: "Advanced", icon: Settings2 },
  { id: "color", label: "Color", icon: Palette },
  { id: "control", label: "Input", icon: Layers },
  { id: "scripts", label: "Scripts", icon: FileCode },
];

/** External links at the bottom of the sidebar */
export const EXTERNAL_LINKS: ExternalLink[] = [
  { label: "Docs", icon: BookOpen, url: "https://vladmandic.github.io/sdnext-docs/" },
  { label: "GitHub", icon: Github, url: "https://github.com/vladmandic/sdnext" },
  { label: "Discord", icon: MessageCircle, url: "https://discord.gg/VjvR2tabEX" },
  { label: "Contributors", icon: Users, url: "https://github.com/vladmandic/sdnext/graphs/contributors" },
];

export const DEFAULT_GENERATION_PARAMS = {
  prompt: "",
  negativePrompt: "",
  sampler: "Euler",
  steps: 20,
  width: 512,
  height: 512,
  batchSize: 1,
  batchCount: 1,
  cfgScale: 7,
  seed: -1,
  denoisingStrength: 0.5,
};

export const ZOOM_LIMITS = { min: 0.1, max: 16 };

export const RESIZE_MODES = ["None", "Fixed", "Crop", "Fill", "Outpaint", "Context aware"];

/** Hires fix resize modes with numeric values for API */
export const HIRES_RESIZE_MODES = [
  { value: "0", label: "None" },
  { value: "1", label: "Fixed" },
  { value: "2", label: "Crop" },
  { value: "3", label: "Fill" },
  { value: "4", label: "Outpaint" },
  { value: "5", label: "Context aware" },
] as const;

/** Context modes for context-aware hires resize */
export const HIRES_CONTEXT_MODES = [
  { value: "None", label: "None" },
  { value: "Add with forward", label: "Add with forward" },
  { value: "Remove with forward", label: "Remove with forward" },
  { value: "Add with backward", label: "Add with backward" },
  { value: "Remove with backward", label: "Remove with backward" },
] as const;
