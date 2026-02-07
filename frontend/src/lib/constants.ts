import type { LucideIcon } from "lucide-react";
import {
  ImageIcon, Video, Sparkles, MessageSquare, Images,
  Type, SlidersHorizontal, Compass, Wand2, ScanSearch,
  Settings2, Cable, Layers, FileCode,
  BookOpen, Github, MessageCircle, Users,
} from "lucide-react";

export interface NavItem {
  id: string;
  label: string;
  icon: LucideIcon;
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

/** Primary sidebar navigation */
export const NAV_ITEMS: NavItem[] = [
  { id: "images", label: "Images", icon: ImageIcon },
  { id: "video", label: "Video", icon: Video },
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
  { id: "adapters", label: "Adapters", icon: Cable },
  { id: "control", label: "Control", icon: Layers },
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
