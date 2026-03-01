import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";
import { base64ToBlob } from "@/lib/utils";
import { createIdbStorage } from "@/lib/idbStorage";
import { useVideoStore } from "@/stores/videoStore";

export interface VideoFrameImage {
  id: string;
  file: File;
  base64: string;
  objectUrl: string;
  naturalWidth: number;
  naturalHeight: number;
}

interface ViewportState {
  x: number;
  y: number;
  scale: number;
}

interface VideoCanvasState {
  viewport: ViewportState;
  initFrame: VideoFrameImage | null;
  lastFrame: VideoFrameImage | null;

  setViewport: (v: Partial<ViewportState>) => void;
  setFrame: (which: "init" | "last", file: File, base64: string, objectUrl: string, w: number, h: number) => void;
  clearFrame: (which: "init" | "last") => void;
  clearAll: () => void;
}

interface PersistedVideoCanvasState {
  viewport: ViewportState;
  initFrame: { id: string; base64: string; naturalWidth: number; naturalHeight: number } | null;
  lastFrame: { id: string; base64: string; naturalWidth: number; naturalHeight: number } | null;
}

const videoCanvasIdbStorage = createIdbStorage("sdnext-video-canvas", "state");

function rehydrateFrame(saved: PersistedVideoCanvasState["initFrame"]): VideoFrameImage | null {
  if (!saved || !saved.base64) return null;
  const blob = base64ToBlob(saved.base64);
  const objectUrl = URL.createObjectURL(blob);
  return {
    id: saved.id,
    file: new File([blob], "restored.png", { type: "image/png" }),
    base64: saved.base64,
    objectUrl,
    naturalWidth: saved.naturalWidth,
    naturalHeight: saved.naturalHeight,
  };
}

export const useVideoCanvasStore = create<VideoCanvasState>()(
  persist(
    (set, get) => ({
      viewport: { x: 0, y: 0, scale: 1 },
      initFrame: null,
      lastFrame: null,

      setViewport: (v) => set((s) => ({ viewport: { ...s.viewport, ...v } })),

      setFrame: (which, file, base64, objectUrl, w, h) => {
        const prev = get()[which === "init" ? "initFrame" : "lastFrame"];
        if (prev?.objectUrl) URL.revokeObjectURL(prev.objectUrl);
        const frame: VideoFrameImage = {
          id: crypto.randomUUID(),
          file, base64, objectUrl,
          naturalWidth: w, naturalHeight: h,
        };
        set({ [which === "init" ? "initFrame" : "lastFrame"]: frame });
      },

      clearFrame: (which) => {
        const key = which === "init" ? "initFrame" : "lastFrame";
        const prev = get()[key];
        if (prev?.objectUrl) URL.revokeObjectURL(prev.objectUrl);
        set({ [key]: null });
      },

      clearAll: () => {
        const { initFrame, lastFrame } = get();
        if (initFrame?.objectUrl) URL.revokeObjectURL(initFrame.objectUrl);
        if (lastFrame?.objectUrl) URL.revokeObjectURL(lastFrame.objectUrl);
        set({ initFrame: null, lastFrame: null });
      },
    }),
    {
      name: "sdnext-video-canvas",
      storage: createJSONStorage(() => videoCanvasIdbStorage),
      partialize: (state): PersistedVideoCanvasState => ({
        viewport: state.viewport,
        initFrame: state.initFrame
          ? { id: state.initFrame.id, base64: state.initFrame.base64, naturalWidth: state.initFrame.naturalWidth, naturalHeight: state.initFrame.naturalHeight }
          : null,
        lastFrame: state.lastFrame
          ? { id: state.lastFrame.id, base64: state.lastFrame.base64, naturalWidth: state.lastFrame.naturalWidth, naturalHeight: state.lastFrame.naturalHeight }
          : null,
      }),
      merge: (persisted, current) => {
        const saved = persisted as Partial<PersistedVideoCanvasState> | undefined;
        if (!saved) return current;
        return {
          ...current,
          viewport: saved.viewport ?? current.viewport,
          initFrame: saved.initFrame ? rehydrateFrame(saved.initFrame) : current.initFrame,
          lastFrame: saved.lastFrame ? rehydrateFrame(saved.lastFrame) : current.lastFrame,
        };
      },
    },
  ),
);

// Sync frames to videoStore so buildJobPayload keeps working
useVideoCanvasStore.subscribe((state, prev) => {
  if (state.initFrame !== prev.initFrame) {
    useVideoStore.getState().setParam("initImage", state.initFrame?.file ?? null);
  }
  if (state.lastFrame !== prev.lastFrame) {
    useVideoStore.getState().setParam("lastImage", state.lastFrame?.file ?? null);
  }
});
