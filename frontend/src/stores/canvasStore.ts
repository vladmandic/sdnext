import { create } from "zustand";

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
  imageData: string;
  x: number;
  y: number;
  width: number;
  height: number;
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
}

export const useCanvasStore = create<CanvasState>()((set) => ({
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

  removeLayer: (id) =>
    set((s) => ({
      layers: s.layers.filter((l) => l.id !== id),
      activeLayerId: s.activeLayerId === id ? null : s.activeLayerId,
    })),

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
  clearLayers: () => set({ layers: [], activeLayerId: null }),
}));
