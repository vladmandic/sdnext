import { create } from "zustand";
import { persist } from "zustand/middleware";

interface ProcessState {
  image: File | null;
  imagePreviewUrl: string | null;
  upscaler: string;
  scale: number;
  resultImageUrl: string | null;
  resultWidth: number | null;
  resultHeight: number | null;
  compareMode: boolean;

  setImage: (file: File | null) => void;
  setUpscaler: (upscaler: string) => void;
  setScale: (scale: number) => void;
  setResult: (url: string | null, width?: number, height?: number) => void;
  setCompareMode: (enabled: boolean) => void;
  reset: () => void;
}

export const useProcessStore = create<ProcessState>()(
  persist(
    (set, get) => ({
      image: null,
      imagePreviewUrl: null,
      upscaler: "None",
      scale: 2,
      resultImageUrl: null,
      resultWidth: null,
      resultHeight: null,
      compareMode: false,

      setImage: (file) => {
        const prev = get().imagePreviewUrl;
        if (prev) URL.revokeObjectURL(prev);
        set({
          image: file,
          imagePreviewUrl: file ? URL.createObjectURL(file) : null,
          resultImageUrl: null,
          resultWidth: null,
          resultHeight: null,
          compareMode: false,
        });
      },

      setUpscaler: (upscaler) => set({ upscaler }),
      setScale: (scale) => set({ scale }),
      setCompareMode: (enabled) => set({ compareMode: enabled }),
      setResult: (url, width, height) => set({
        resultImageUrl: url,
        resultWidth: width ?? null,
        resultHeight: height ?? null,
      }),

      reset: () => {
        const prev = get().imagePreviewUrl;
        if (prev) URL.revokeObjectURL(prev);
        set({
          image: null,
          imagePreviewUrl: null,
          upscaler: "None",
          scale: 2,
          resultImageUrl: null,
          resultWidth: null,
          resultHeight: null,
        });
      },
    }),
    {
      name: "sdnext-process",
      partialize: ({ upscaler, scale }) => ({ upscaler, scale }),
    },
  ),
);
