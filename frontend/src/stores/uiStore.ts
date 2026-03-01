import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { AsideTab } from "@/lib/constants";

type SidebarView = "images" | "video" | "process" | "caption" | "gallery";
type ImagesSubTab = "prompts" | "sampler" | "guidance" | "refine" | "detail" | "advanced" | "control" | "scripts";
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

  // Aside tabs
  activeAsideTab: AsideTab;

  // Result gallery
  resultThumbSize: number;

  // Canvas preferences
  autoFitFrame: boolean;
  reprocessOnGenerate: boolean;

  // Model defaults
  autoApplyModelDefaults: boolean;

  // Command palette
  recentCommandIds: string[];

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
  setResultThumbSize: (size: number) => void;
  setAutoFitFrame: (enabled: boolean) => void;
  setAutoUpdateProcessed: (enabled: boolean) => void;
  setAutoApplyModelDefaults: (enabled: boolean) => void;
  toggleLeftPanel: () => void;
  setLeftPanelWidth: (width: number) => void;
  toggleRightPanel: () => void;
  setAsideTab: (tab: AsideTab) => void;
  openAsideTab: (tab: AsideTab) => void;
  setColorMode: (mode: ColorMode) => void;
  setAccentColor: (color: string) => void;
  setCornerStyle: (style: CornerStyle) => void;
  setBorderRadius: (radius: number) => void;
  setUiScale: (scale: number) => void;
  setCanvasLabelScale: (scale: number) => void;
  addRecentCommand: (id: string) => void;
}

export type { SidebarView, ImagesSubTab, CornerStyle, ColorMode };

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
      activeAsideTab: "networks" as AsideTab,
      resultThumbSize: 56,
      autoFitFrame: true,
      reprocessOnGenerate: true,
      autoApplyModelDefaults: false,
      recentCommandIds: [],
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
      setResultThumbSize: (size) => set({ resultThumbSize: Math.max(40, Math.min(160, size)) }),
      setAutoFitFrame: (enabled) => set({ autoFitFrame: enabled }),
      setAutoUpdateProcessed: (enabled) => set({ reprocessOnGenerate: enabled }),
      setAutoApplyModelDefaults: (enabled) => set({ autoApplyModelDefaults: enabled }),
      toggleLeftPanel: () => set((s) => ({ leftPanelCollapsed: !s.leftPanelCollapsed })),
      setLeftPanelWidth: (width) => set({ leftPanelWidth: Math.max(280, Math.min(600, width)) }),
      toggleRightPanel: () => set((s) => ({ rightPanelCollapsed: !s.rightPanelCollapsed })),
      setAsideTab: (tab) => set({ activeAsideTab: tab }),
      openAsideTab: (tab) => set({ activeAsideTab: tab, rightPanelCollapsed: false }),
      setColorMode: (mode) => set({ colorMode: mode }),
      setAccentColor: (color) => set({ accentColor: color }),
      setCornerStyle: (style) => set({ cornerStyle: style }),
      setBorderRadius: (radius) => set({ borderRadius: Math.max(0, Math.min(1, radius)) }),
      setUiScale: (scale) => set({ uiScale: Math.max(12, Math.min(20, scale)) }),
      setCanvasLabelScale: (scale) => set({ canvasLabelScale: Math.max(0.5, Math.min(2, scale)) }),
      addRecentCommand: (id) => set((s) => {
        const filtered = s.recentCommandIds.filter((c) => c !== id);
        return { recentCommandIds: [id, ...filtered].slice(0, 5) };
      }),
    }),
    { name: "sdnext-ui-v2" },
  ),
);
