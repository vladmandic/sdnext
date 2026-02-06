import { create } from "zustand";
import { persist } from "zustand/middleware";

type SidebarView = "images" | "video" | "process" | "caption" | "gallery";
type ImagesSubTab = "prompts" | "sampler" | "guidance" | "refine" | "detail" | "advanced" | "adapters" | "control" | "scripts";

interface UiState {
  // Sidebar
  sidebarCollapsed: boolean;
  activeSidebarView: SidebarView;
  activeImagesSubTab: ImagesSubTab;

  // Panels
  leftPanelCollapsed: boolean;
  leftPanelWidth: number;
  rightPanelCollapsed: boolean;
  rightPanelWidth: number;

  // Theme
  theme: string;

  // Actions
  toggleSidebar: () => void;
  setSidebarView: (view: SidebarView) => void;
  setImagesSubTab: (tab: ImagesSubTab) => void;
  toggleLeftPanel: () => void;
  setLeftPanelWidth: (width: number) => void;
  toggleRightPanel: () => void;
  setRightPanelWidth: (width: number) => void;
  setTheme: (theme: string) => void;
}

export type { SidebarView, ImagesSubTab };

export const useUiStore = create<UiState>()(
  persist(
    (set) => ({
      sidebarCollapsed: false,
      activeSidebarView: "images",
      activeImagesSubTab: "prompts",
      leftPanelCollapsed: false,
      leftPanelWidth: 380,
      rightPanelCollapsed: true,
      rightPanelWidth: 320,
      theme: "default",

      toggleSidebar: () => set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed })),
      setSidebarView: (view) => set({ activeSidebarView: view }),
      setImagesSubTab: (tab) => set({ activeImagesSubTab: tab }),
      toggleLeftPanel: () => set((s) => ({ leftPanelCollapsed: !s.leftPanelCollapsed })),
      setLeftPanelWidth: (width) => set({ leftPanelWidth: Math.max(280, Math.min(600, width)) }),
      toggleRightPanel: () => set((s) => ({ rightPanelCollapsed: !s.rightPanelCollapsed })),
      setRightPanelWidth: (width) => set({ rightPanelWidth: Math.max(280, Math.min(600, width)) }),
      setTheme: (theme) => set({ theme }),
    }),
    { name: "sdnext-ui-v2" },
  ),
);
