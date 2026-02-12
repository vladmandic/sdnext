import { create } from "zustand";
import type { CaptionMethod, OpenClipResponse, TaggerResponse, VqaResponse } from "@/api/types/caption";

export type CaptionResult =
  | (OpenClipResponse & { type: "openclip" })
  | (TaggerResponse & { type: "tagger" })
  | (VqaResponse & { type: "vqa" })
  | null;

interface CaptionState {
  image: File | null;
  imagePreviewUrl: string | null;
  result: CaptionResult;
  isProcessing: boolean;
  method: CaptionMethod;

  setImage: (file: File | null) => void;
  setResult: (result: CaptionResult) => void;
  setProcessing: (v: boolean) => void;
  setMethod: (m: CaptionMethod) => void;
  reset: () => void;
}

export const useCaptionStore = create<CaptionState>()((set, get) => ({
  image: null,
  imagePreviewUrl: null,
  result: null,
  isProcessing: false,
  method: "vlm",

  setImage: (file) => {
    const prev = get().imagePreviewUrl;
    if (prev) URL.revokeObjectURL(prev);
    set({
      image: file,
      imagePreviewUrl: file ? URL.createObjectURL(file) : null,
      result: null,
    });
  },

  setResult: (result) => set({ result }),
  setProcessing: (v) => set({ isProcessing: v }),
  setMethod: (m) => set({ method: m }),

  reset: () => {
    const prev = get().imagePreviewUrl;
    if (prev) URL.revokeObjectURL(prev);
    set({
      image: null,
      imagePreviewUrl: null,
      result: null,
      isProcessing: false,
      method: "vlm",
    });
  },
}));
