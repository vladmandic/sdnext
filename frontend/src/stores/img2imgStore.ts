import { create } from "zustand";
import { useCanvasStore } from "@/stores/canvasStore";
import type { SizeMode } from "@/lib/sizeCompute";

export type { SizeMode };

export interface MaskLine {
  points: number[];       // flat [x1,y1,x2,y2,...] in image-space pixels
  strokeWidth: number;
  tool: "brush" | "eraser";
}

interface Img2ImgState {
  // Resize mode
  resizeMode: number;

  // Size mode (Fixed / Scale / Megapixel)
  sizeMode: SizeMode;
  scaleFactor: number;
  megapixelTarget: number;

  // Mask painting
  maskLines: MaskLine[];

  // Mask params
  maskData: string | null;
  maskBlur: number;
  inpaintFullRes: boolean;
  inpaintFullResPadding: number;
  inpaintingMaskInvert: boolean;

  // Actions
  addMaskLine: (line: MaskLine) => void;
  clearMask: () => void;
  setResizeMode: (mode: number) => void;
  setSizeMode: (mode: SizeMode) => void;
  setScaleFactor: (factor: number) => void;
  setMegapixelTarget: (target: number) => void;
  setMaskBlur: (blur: number) => void;
  setInpaintFullRes: (v: boolean) => void;
  setInpaintFullResPadding: (v: number) => void;
  setInpaintingMaskInvert: (v: boolean) => void;
  hasLayers: () => boolean;
  reset: () => void;
}

const defaultState = {
  resizeMode: 1,
  sizeMode: "fixed" as SizeMode,
  scaleFactor: 1,
  megapixelTarget: 1,
  maskLines: [] as MaskLine[],
  maskData: null as string | null,
  maskBlur: 4,
  inpaintFullRes: false,
  inpaintFullResPadding: 32,
  inpaintingMaskInvert: false,
};

export const useImg2ImgStore = create<Img2ImgState>()((_set) => ({
  ...defaultState,

  addMaskLine: (line) => _set((s) => ({ maskLines: [...s.maskLines, line] })),

  clearMask: () => _set({ maskLines: [], maskData: null }),

  setResizeMode: (mode) => _set({ resizeMode: mode }),
  setSizeMode: (mode) => _set({ sizeMode: mode }),
  setScaleFactor: (factor) => _set({ scaleFactor: factor }),
  setMegapixelTarget: (target) => _set({ megapixelTarget: target }),
  setMaskBlur: (blur) => _set({ maskBlur: blur }),
  setInpaintFullRes: (v) => _set({ inpaintFullRes: v }),
  setInpaintFullResPadding: (v) => _set({ inpaintFullResPadding: v }),
  setInpaintingMaskInvert: (v) => _set({ inpaintingMaskInvert: v }),

  hasLayers: () => useCanvasStore.getState().layers.length > 0,

  reset: () => _set(defaultState),
}));
