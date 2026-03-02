import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";
import { useGenerationStore } from "@/stores/generationStore";
import { useUiStore } from "@/stores/uiStore";
import { base64ToBlob } from "@/lib/utils";
import { createIdbStorage } from "@/lib/idbStorage";

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

export interface MaskObjectLayer extends CanvasLayer {
  type: "mask";
  imageData: string;       // object URL of colored display image
  base64: string;          // colored PNG base64 for persistence
  x: number;
  y: number;
  width: number;
  height: number;
  scaleX: number;
  scaleY: number;
  rotation: number;
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
  inputRole: "initial" | "reference";
  selectedControlFrame: number | null;
  panelCollapsedOverrides: Map<number, boolean>;  // explicit user overrides

  setInputRole: (role: "initial" | "reference") => void;
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
  setSelectedControlFrame: (index: number | null) => void;
  togglePanelCollapsed: (index: number, currentCollapsed: boolean) => void;
  clearLayers: () => void;
  restoreImageLayer: (base64: string, w: number, h: number) => void;
  getImageLayers: () => ImageLayer[];
  getMaskLayers: () => MaskObjectLayer[];
  replaceMaskLayers: (newLayers: MaskObjectLayer[]) => void;
  removeMaskLayers: () => void;
}

/** Serializable snapshot of canvas state stored in IndexedDB. */
interface PersistedCanvasState {
  viewport: ViewportState;
  layers: CanvasLayer[];
  activeLayerId: string | null;
  activeTool: ToolType;
  inputRole: "initial" | "reference";
  brushSize: number;
  brushHardness: number;
  brushColor: string;
  brushOpacity: number;
  maskVisible: boolean;
  maskColor: string;
  panelCollapsedOverrides: [number, boolean][];
}

const canvasIdbStorage = createIdbStorage("sdnext-canvas", "state");

function rehydrateLayer(layer: CanvasLayer): CanvasLayer | ImageLayer | MaskObjectLayer {
  if (layer.type === "image") {
    const img = layer as ImageLayer;
    if (!img.base64) return layer;
    const blob = base64ToBlob(img.base64);
    return {
      ...img,
      imageData: URL.createObjectURL(blob),
      file: new File([blob], img.name || "restored.png", { type: "image/png" }),
    };
  }
  if (layer.type === "mask") {
    const ml = layer as MaskObjectLayer;
    if (!ml.base64) return layer;
    const blob = base64ToBlob(ml.base64);
    return { ...ml, imageData: URL.createObjectURL(blob) };
  }
  return layer;
}

export const useCanvasStore = create<CanvasState>()(
  persist(
    (set, get) => ({
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
      inputRole: "initial",
      selectedControlFrame: null,
      panelCollapsedOverrides: new Map<number, boolean>(),

      setInputRole: (role) => set({ inputRole: role }),
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
          if (layer && (layer.type === "image" || layer.type === "mask")) {
            URL.revokeObjectURL((layer as ImageLayer | MaskObjectLayer).imageData);
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
      setSelectedControlFrame: (index) => set({ selectedControlFrame: index }),

      togglePanelCollapsed: (index, currentCollapsed: boolean) => set((s) => {
        const newMap = new Map(s.panelCollapsedOverrides);
        newMap.set(index, !currentCollapsed);
        return { panelCollapsedOverrides: newMap };
      }),

      clearLayers: () => {
        const { layers } = get();
        for (const layer of layers) {
          if (layer.type === "image" || layer.type === "mask") {
            URL.revokeObjectURL((layer as ImageLayer | MaskObjectLayer).imageData);
          }
        }
        set({ layers: [], activeLayerId: null });
      },

      restoreImageLayer: (base64, w, h) => {
        // Clear existing layers first
        const { layers } = get();
        for (const layer of layers) {
          if (layer.type === "image") {
            URL.revokeObjectURL((layer as ImageLayer).imageData);
          }
        }
        const blob = base64ToBlob(base64);
        const objectUrl = URL.createObjectURL(blob);
        const id = crypto.randomUUID();
        const layer: ImageLayer = {
          id,
          type: "image",
          name: "Restored input",
          visible: true,
          opacity: 1,
          locked: false,
          imageData: objectUrl,
          base64,
          file: new File([blob], "restored.png", { type: "image/png" }),
          naturalWidth: w,
          naturalHeight: h,
          x: 0,
          y: 0,
          width: w,
          height: h,
          rotation: 0,
          scaleX: 1,
          scaleY: 1,
        };
        set({ layers: [layer], activeLayerId: id });
      },

      getImageLayers: () => get().layers.filter((l) => l.type === "image") as ImageLayer[],

      getMaskLayers: () => get().layers.filter((l) => l.type === "mask") as MaskObjectLayer[],

      replaceMaskLayers: (newLayers) => {
        const { layers } = get();
        for (const l of layers) {
          if (l.type === "mask") URL.revokeObjectURL((l as MaskObjectLayer).imageData);
        }
        set((s) => ({
          layers: [...s.layers.filter((l) => l.type !== "mask"), ...newLayers],
        }));
      },

      removeMaskLayers: () => {
        const { layers } = get();
        for (const l of layers) {
          if (l.type === "mask") URL.revokeObjectURL((l as MaskObjectLayer).imageData);
        }
        set((s) => ({
          layers: s.layers.filter((l) => l.type !== "mask"),
          activeLayerId: s.activeLayerId && s.layers.find((l) => l.id === s.activeLayerId)?.type === "mask" ? null : s.activeLayerId,
        }));
      },
    }),
    {
      name: "sdnext-canvas",
      storage: createJSONStorage(() => canvasIdbStorage),
      partialize: (state): PersistedCanvasState => ({
        viewport: state.viewport,
        inputRole: state.inputRole,
        layers: state.layers.map((layer) => {
          if (layer.type === "image") {
            const { file: _file, imageData: _url, ...rest } = layer as ImageLayer;
            return { ...rest, imageData: "", file: undefined } as unknown as CanvasLayer;
          }
          if (layer.type === "mask") {
            const { imageData: _url, ...rest } = layer as MaskObjectLayer;
            return { ...rest, imageData: "" } as unknown as CanvasLayer;
          }
          return layer;
        }),
        activeLayerId: state.activeLayerId,
        activeTool: state.activeTool,
        brushSize: state.brushSize,
        brushHardness: state.brushHardness,
        brushColor: state.brushColor,
        brushOpacity: state.brushOpacity,
        maskVisible: state.maskVisible,
        maskColor: state.maskColor,
        panelCollapsedOverrides: [...state.panelCollapsedOverrides.entries()],
      }),
      merge: (persisted, current) => {
        const saved = persisted as Partial<PersistedCanvasState> | undefined;
        if (!saved) return current;
        return {
          ...current,
          viewport: saved.viewport ?? current.viewport,
          activeLayerId: saved.activeLayerId ?? current.activeLayerId,
          activeTool: saved.activeTool ?? current.activeTool,
          inputRole: saved.inputRole ?? "initial",
          brushSize: saved.brushSize ?? current.brushSize,
          brushHardness: saved.brushHardness ?? current.brushHardness,
          brushColor: saved.brushColor ?? current.brushColor,
          brushOpacity: saved.brushOpacity ?? current.brushOpacity,
          maskVisible: saved.maskVisible ?? current.maskVisible,
          maskColor: saved.maskColor ?? current.maskColor,
          panelCollapsedOverrides: saved.panelCollapsedOverrides
            ? new Map(saved.panelCollapsedOverrides)
            : current.panelCollapsedOverrides,
          layers: saved.layers ? saved.layers.map(rehydrateLayer) : current.layers,
        };
      },
    },
  ),
);
