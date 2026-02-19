import { create } from "zustand";

interface ProcessState {
  image: File | null;
  imagePreviewUrl: string | null;
  upscaler: string;
  scale: number;
  isProcessing: boolean;
  jobId: string | null;
  resultImageUrl: string | null;
  resultWidth: number | null;
  resultHeight: number | null;

  setImage: (file: File | null) => void;
  setUpscaler: (upscaler: string) => void;
  setScale: (scale: number) => void;
  setProcessing: (processing: boolean) => void;
  setJobId: (id: string | null) => void;
  setResult: (url: string | null, width?: number, height?: number) => void;
  reset: () => void;
}

export const useProcessStore = create<ProcessState>()((set, get) => ({
  image: null,
  imagePreviewUrl: null,
  upscaler: "None",
  scale: 2,
  isProcessing: false,
  jobId: null,
  resultImageUrl: null,
  resultWidth: null,
  resultHeight: null,

  setImage: (file) => {
    const prev = get().imagePreviewUrl;
    if (prev) URL.revokeObjectURL(prev);
    set({
      image: file,
      imagePreviewUrl: file ? URL.createObjectURL(file) : null,
      resultImageUrl: null,
      resultWidth: null,
      resultHeight: null,
    });
  },

  setUpscaler: (upscaler) => set({ upscaler }),
  setScale: (scale) => set({ scale }),
  setProcessing: (processing) => set({ isProcessing: processing }),
  setJobId: (id) => set({ jobId: id }),

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
      isProcessing: false,
      jobId: null,
      resultImageUrl: null,
      resultWidth: null,
      resultHeight: null,
    });
  },
}));
