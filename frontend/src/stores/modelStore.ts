import { create } from "zustand";

interface ModelState {
  currentModel: string | null;
  currentVae: string | null;
  isModelLoading: boolean;

  setCurrentModel: (model: string | null) => void;
  setCurrentVae: (vae: string | null) => void;
  setModelLoading: (loading: boolean) => void;
}

export const useModelStore = create<ModelState>()((set) => ({
  currentModel: null,
  currentVae: null,
  isModelLoading: false,

  setCurrentModel: (model) => set({ currentModel: model }),
  setCurrentVae: (vae) => set({ currentVae: vae }),
  setModelLoading: (loading) => set({ isModelLoading: loading }),
}));
