import { create } from "zustand";
import { useGenerationStore } from "@/stores/generationStore";
import { useUiStore } from "@/stores/uiStore";

export type ToolType = "move" | "brush" | "eraser" | "maskBrush" | "maskEraser" | "rectSelect" | "lassoSelect" | "colorPicker" | "zoom" | "pan";

export interface CanvasLayer {
  id: string;
  type: "image" | "drawing" | "mask" | "generation";
  visible: boolean;
  opacity: number;
  locked: boolean;
  name: string;
}

export interface ImageLayer extends CanvasLayer {
  type: "image";
  imageData: string;       // object URL for Konva display
  base64: string;          // raw base64 for flattening / API
  file: File;              // original File object
  naturalWidth: number;    // original image pixel width
  naturalHeight: number;   // original image pixel height
  x: number;
  y: number;
  width: number;           // = naturalWidth (display reference)
  height: number;          // = naturalHeight (display reference)
  rotation: number;
  scaleX: number;
  scaleY: number;
}

interface ViewportState {
  x: number;
  y: number;
  scale: number;
}

interface CanvasState {
  viewport: ViewportState;
  layers: CanvasLayer[];
  activeLayerId: string | null;
  activeTool: ToolType;
  brushSize: number;
  brushHardness: number;
  brushColor: string;
  brushOpacity: number;
  selection: { x: number; y: number; width: number; height: number } | null;
  maskVisible: boolean;
  maskColor: string;

  setViewport: (viewport: Partial<ViewportState>) => void;
  addLayer: (layer: CanvasLayer) => void;
  addImageLayer: (file: File, base64: string, objectUrl: string, w: number, h: number) => void;
  removeLayer: (id: string) => void;
  updateLayer: (id: string, updates: Partial<CanvasLayer>) => void;
  setActiveLayer: (id: string | null) => void;
  setActiveTool: (tool: ToolType) => void;
  setBrushSize: (size: number) => void;
  setBrushColor: (color: string) => void;
  setBrushOpacity: (opacity: number) => void;
  setSelection: (rect: CanvasState["selection"]) => void;
  setMaskVisible: (visible: boolean) => void;
  setMaskColor: (color: string) => void;
  clearLayers: () => void;
  getImageLayers: () => ImageLayer[];
}

export const useCanvasStore = create<CanvasState>()((set, get) => ({
  viewport: { x: 0, y: 0, scale: 1 },
  layers: [],
  activeLayerId: null,
  activeTool: "move",
  brushSize: 20,
  brushHardness: 0.8,
  brushColor: "#ffffff",
  brushOpacity: 1,
  selection: null,
  maskVisible: true,
  maskColor: "#ff000080",

  setViewport: (viewport) =>
    set((s) => ({ viewport: { ...s.viewport, ...viewport } })),

  addLayer: (layer) =>
    set((s) => ({ layers: [...s.layers, layer] })),

  addImageLayer: (file, base64, objectUrl, w, h) => {
    const gen = useGenerationStore.getState();
    const { layers } = get();

    // Auto-resize frame to match first image (snap to 8px grid)
    const autoFit = useUiStore.getState().autoFitFrame;
    if (layers.length === 0 && autoFit) {
      const snapW = Math.round(w / 8) * 8;
      const snapH = Math.round(h / 8) * 8;
      gen.setParam("width", snapW);
      gen.setParam("height", snapH);
    }

    // Use (potentially just-updated) frame dimensions for centering
    const frameW = layers.length === 0 && autoFit ? Math.round(w / 8) * 8 : gen.width;
    const frameH = layers.length === 0 && autoFit ? Math.round(h / 8) * 8 : gen.height;

    const id = crypto.randomUUID();
    const layer: ImageLayer = {
      id,
      type: "image",
      name: file.name,
      visible: true,
      opacity: 1,
      locked: false,
      imageData: objectUrl,
      base64,
      file,
      naturalWidth: w,
      naturalHeight: h,
      x: Math.round((frameW - w) / 2),
      y: Math.round((frameH - h) / 2),
      width: w,
      height: h,
      rotation: 0,
      scaleX: 1,
      scaleY: 1,
    };
    set((s) => ({ layers: [...s.layers, layer], activeLayerId: id }));
  },

  removeLayer: (id) =>
    set((s) => {
      const layer = s.layers.find((l) => l.id === id);
      if (layer && layer.type === "image") {
        URL.revokeObjectURL((layer as ImageLayer).imageData);
      }
      return {
        layers: s.layers.filter((l) => l.id !== id),
        activeLayerId: s.activeLayerId === id ? null : s.activeLayerId,
      };
    }),

  updateLayer: (id, updates) =>
    set((s) => ({
      layers: s.layers.map((l) => (l.id === id ? { ...l, ...updates } : l)),
    })),

  setActiveLayer: (id) => set({ activeLayerId: id }),
  setActiveTool: (tool) => set({ activeTool: tool }),
  setBrushSize: (size) => set({ brushSize: size }),
  setBrushColor: (color) => set({ brushColor: color }),
  setBrushOpacity: (opacity) => set({ brushOpacity: opacity }),
  setSelection: (rect) => set({ selection: rect }),
  setMaskVisible: (visible) => set({ maskVisible: visible }),
  setMaskColor: (color) => set({ maskColor: color }),

  clearLayers: () => {
    const { layers } = get();
    for (const layer of layers) {
      if (layer.type === "image") {
        URL.revokeObjectURL((layer as ImageLayer).imageData);
      }
    }
    set({ layers: [], activeLayerId: null });
  },

  getImageLayers: () => get().layers.filter((l) => l.type === "image") as ImageLayer[],
}));
