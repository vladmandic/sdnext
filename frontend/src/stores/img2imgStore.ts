import { create } from "zustand";
import { fileToBase64 } from "@/lib/image";
import { useGenerationStore } from "@/stores/generationStore";

function snap8(v: number): number {
  return Math.round(v / 8) * 8;
}

export interface MaskLine {
  points: number[];       // flat [x1,y1,x2,y2,...] in image-space pixels
  strokeWidth: number;
  tool: "brush" | "eraser";
}

interface Img2ImgState {
  // Init image
  initImageData: string | null;
  initImageBase64: string | null;
  initImageWidth: number;
  initImageHeight: number;
  initImageFile: File | null;
  initImageName: string;

  // Resolution mode
  resolutionMode: "auto" | "custom";
  resizeMode: number;

  // Mask painting
  maskLines: MaskLine[];

  // Mask params
  maskData: string | null;
  maskBlur: number;
  inpaintFullRes: boolean;
  inpaintFullResPadding: number;
  inpaintingMaskInvert: boolean;

  // Actions
  setInitImage: (file: File) => Promise<void>;
  clearInitImage: () => void;
  addMaskLine: (line: MaskLine) => void;
  clearMask: () => void;
  setResolutionMode: (mode: "auto" | "custom") => void;
  setResizeMode: (mode: number) => void;
  setMaskBlur: (blur: number) => void;
  setInpaintFullRes: (v: boolean) => void;
  setInpaintFullResPadding: (v: number) => void;
  setInpaintingMaskInvert: (v: boolean) => void;
  reset: () => void;
}

const defaultState = {
  initImageData: null,
  initImageBase64: null,
  initImageWidth: 0,
  initImageHeight: 0,
  initImageFile: null,
  initImageName: "",
  resolutionMode: "auto" as const,
  resizeMode: 1,
  maskLines: [] as MaskLine[],
  maskData: null,
  maskBlur: 4,
  inpaintFullRes: false,
  inpaintFullResPadding: 32,
  inpaintingMaskInvert: false,
};

export const useImg2ImgStore = create<Img2ImgState>()((set, get) => ({
  ...defaultState,

  setInitImage: async (file: File) => {
    // Revoke previous object URL
    const prev = get().initImageData;
    if (prev) URL.revokeObjectURL(prev);

    const base64 = await fileToBase64(file);
    const objectUrl = URL.createObjectURL(file);

    // Load image to get natural dimensions
    const img = new window.Image();
    img.src = objectUrl;
    await new Promise<void>((resolve) => { img.onload = () => resolve(); });

    const w = img.naturalWidth;
    const h = img.naturalHeight;

    set({
      initImageBase64: base64,
      initImageData: objectUrl,
      initImageWidth: w,
      initImageHeight: h,
      initImageFile: file,
      initImageName: file.name,
    });

    // Auto-match resolution
    if (get().resolutionMode === "auto") {
      useGenerationStore.getState().setParams({
        width: snap8(w),
        height: snap8(h),
      });
    }
  },

  clearInitImage: () => {
    const prev = get().initImageData;
    if (prev) URL.revokeObjectURL(prev);
    set({
      initImageData: null,
      initImageBase64: null,
      initImageWidth: 0,
      initImageHeight: 0,
      initImageFile: null,
      initImageName: "",
      maskLines: [],
      maskData: null,
    });
  },

  addMaskLine: (line) => set((s) => ({ maskLines: [...s.maskLines, line] })),

  clearMask: () => set({ maskLines: [], maskData: null }),

  setResolutionMode: (mode) => {
    set({ resolutionMode: mode });
    if (mode === "auto") {
      const { initImageWidth, initImageHeight } = get();
      if (initImageWidth > 0 && initImageHeight > 0) {
        useGenerationStore.getState().setParams({
          width: snap8(initImageWidth),
          height: snap8(initImageHeight),
        });
      }
    }
  },

  setResizeMode: (mode) => set({ resizeMode: mode }),
  setMaskBlur: (blur) => set({ maskBlur: blur }),
  setInpaintFullRes: (v) => set({ inpaintFullRes: v }),
  setInpaintFullResPadding: (v) => set({ inpaintFullResPadding: v }),
  setInpaintingMaskInvert: (v) => set({ inpaintingMaskInvert: v }),

  reset: () => {
    const prev = get().initImageData;
    if (prev) URL.revokeObjectURL(prev);
    set(defaultState);
  },
}));
