import { create } from "zustand";
import { persist } from "zustand/middleware";
import { useUiStore } from "@/stores/uiStore";

export interface TutorialStep {
  target: string;
  title: string;
  description: string;
  placement: "top" | "right" | "bottom" | "left";
}

export const TUTORIAL_STEPS: TutorialStep[] = [
  {
    target: "prompt-editor",
    title: "Enter a prompt here",
    description: "Describe what you want to generate. Use natural language or keywords — the more detail, the better the result.",
    placement: "right",
  },
  {
    target: "generate-button",
    title: "Click Generate",
    description: "Once your prompt is ready, hit Generate to create images. You can also press Ctrl+Enter from anywhere.",
    placement: "right",
  },
  {
    target: "result-gallery",
    title: "Your results appear here",
    description: "Generated images show up in this gallery. Click a thumbnail to select it, double-click to restore its settings.",
    placement: "top",
  },
  {
    target: "sidebar-subtabs",
    title: "Explore more options",
    description: "These tabs let you fine-tune sampler settings, guidance, refinement, control networks, and more.",
    placement: "right",
  },
];

interface TutorialState {
  active: boolean;
  currentStep: number;
  completed: boolean;

  start: () => void;
  next: () => void;
  back: () => void;
  skip: () => void;
  reset: () => void;
}

export const useTutorialStore = create<TutorialState>()(
  persist(
    (set) => ({
      active: false,
      currentStep: 0,
      completed: false,

      start: () => {
        const ui = useUiStore.getState();
        if (ui.sidebarCollapsed) ui.toggleSidebar();
        ui.setSidebarView("images");
        ui.setImagesSubTab("prompts");
        if (ui.leftPanelCollapsed) ui.toggleLeftPanel();
        if (ui.viewCollapsed) ui.toggleViewCollapsed();
        set({ active: true, currentStep: 0 });
      },

      next: () => set((s) => {
        if (s.currentStep >= TUTORIAL_STEPS.length - 1) {
          return { active: false, completed: true };
        }
        return { currentStep: s.currentStep + 1 };
      }),

      back: () => set((s) => {
        if (s.currentStep <= 0) return s;
        return { currentStep: s.currentStep - 1 };
      }),

      skip: () => set({ active: false, completed: true }),

      reset: () => set({ completed: false }),
    }),
    {
      name: "sdnext-tutorial",
      partialize: (state) => ({ completed: state.completed }),
    },
  ),
);
