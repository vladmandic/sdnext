import { create } from "zustand";
import { persist } from "zustand/middleware";

type SidebarView = "images" | "video" | "process" | "caption" | "gallery";
type ImagesSubTab = "prompts" | "sampler" | "guidance" | "refine" | "detail" | "advanced" | "control" | "scripts";
type GenerationMode = "txt2img" | "img2img";
type CornerStyle = "rounded" | "square";
type ColorMode = "dark" | "light" | "system";

interface UiState {
  // Sidebar
  sidebarCollapsed: boolean;
  activeSidebarView: SidebarView;
  activeImagesSubTab: ImagesSubTab;
  viewCollapsed: boolean;

  // Panels
  leftPanelCollapsed: boolean;
  leftPanelWidth: number;
  rightPanelCollapsed: boolean;
  rightPanelWidth: number;

  // Generation mode
  generationMode: GenerationMode;

  // Canvas preferences
  autoFitFrame: boolean;

  // Appearance
  colorMode: ColorMode;
  accentColor: string;
  cornerStyle: CornerStyle;
  borderRadius: number;
  uiScale: number;
  canvasLabelScale: number;

  // Actions
  toggleSidebar: () => void;
  setSidebarView: (view: SidebarView) => void;
  setImagesSubTab: (tab: ImagesSubTab) => void;
  toggleViewCollapsed: () => void;
  setGenerationMode: (mode: GenerationMode) => void;
  setAutoFitFrame: (enabled: boolean) => void;
  toggleLeftPanel: () => void;
  setLeftPanelWidth: (width: number) => void;
  toggleRightPanel: () => void;
  setRightPanelWidth: (width: number) => void;
  setColorMode: (mode: ColorMode) => void;
  setAccentColor: (color: string) => void;
  setCornerStyle: (style: CornerStyle) => void;
  setBorderRadius: (radius: number) => void;
  setUiScale: (scale: number) => void;
  setCanvasLabelScale: (scale: number) => void;
}

export type { SidebarView, ImagesSubTab, GenerationMode, CornerStyle, ColorMode };

export const useUiStore = create<UiState>()(
  persist(
    (set) => ({
      sidebarCollapsed: false,
      activeSidebarView: "images",
      activeImagesSubTab: "prompts",
      viewCollapsed: false,
      leftPanelCollapsed: false,
      leftPanelWidth: 380,
      rightPanelCollapsed: true,
      rightPanelWidth: 320,
      generationMode: "txt2img" as GenerationMode,
      autoFitFrame: true,
      colorMode: "dark" as ColorMode,
      accentColor: "#00bcd4",
      cornerStyle: "rounded" as CornerStyle,
      borderRadius: 0.5,
      uiScale: 16,
      canvasLabelScale: 1,

      toggleSidebar: () => set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed })),
      setSidebarView: (view) => set({ activeSidebarView: view }),
      setImagesSubTab: (tab) => set({ activeImagesSubTab: tab }),
      toggleViewCollapsed: () => set((s) => ({ viewCollapsed: !s.viewCollapsed })),
      setGenerationMode: (mode) => set({ generationMode: mode }),
      setAutoFitFrame: (enabled) => set({ autoFitFrame: enabled }),
      toggleLeftPanel: () => set((s) => ({ leftPanelCollapsed: !s.leftPanelCollapsed })),
      setLeftPanelWidth: (width) => set({ leftPanelWidth: Math.max(280, Math.min(600, width)) }),
      toggleRightPanel: () => set((s) => ({ rightPanelCollapsed: !s.rightPanelCollapsed })),
      setRightPanelWidth: (width) => set({ rightPanelWidth: Math.max(280, Math.min(600, width)) }),
      setColorMode: (mode) => set({ colorMode: mode }),
      setAccentColor: (color) => set({ accentColor: color }),
      setCornerStyle: (style) => set({ cornerStyle: style }),
      setBorderRadius: (radius) => set({ borderRadius: Math.max(0, Math.min(1, radius)) }),
      setUiScale: (scale) => set({ uiScale: Math.max(12, Math.min(20, scale)) }),
      setCanvasLabelScale: (scale) => set({ canvasLabelScale: Math.max(0.5, Math.min(2, scale)) }),
    }),
    { name: "sdnext-ui-v2" },
  ),
);
