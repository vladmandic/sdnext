import { create } from "zustand";

type ComparisonMode = "side-by-side" | "swipe" | "overlay" | "diff";

interface ComparisonImage {
  src: string;
  label: string;
  resultId?: string;
  imageIndex?: number;
}

interface ComparisonState {
  open: boolean;
  mode: ComparisonMode;
  imageA: ComparisonImage | null;
  imageB: ComparisonImage | null;
  openComparison: (a: ComparisonImage, b: ComparisonImage, mode?: ComparisonMode) => void;
  closeComparison: () => void;
  setMode: (mode: ComparisonMode) => void;
  swapImages: () => void;
}

export type { ComparisonMode, ComparisonImage };

export const useComparisonStore = create<ComparisonState>()((set) => ({
  open: false,
  mode: "side-by-side",
  imageA: null,
  imageB: null,

  openComparison: (a, b, mode) => set({ open: true, imageA: a, imageB: b, mode: mode ?? "side-by-side" }),
  closeComparison: () => set({ open: false, imageA: null, imageB: null }),
  setMode: (mode) => set({ mode }),
  swapImages: () => set((s) => ({ imageA: s.imageB, imageB: s.imageA })),
}));
