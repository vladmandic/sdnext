import { create } from "zustand";

export interface BackendStatus {
  status: string;
  task: string;
  textinfo: string | null;
  current: string;
  step: number;
  steps: number;
  progress: number;
  eta: number | null;
  elapsed: number | null;
  uptime: number;
  connected: boolean;
  previewUrl: string | null;
}

interface BackendStatusState extends BackendStatus {
  setStatus: (data: Record<string, unknown>) => void;
  setPreview: (url: string) => void;
  setConnected: (connected: boolean) => void;
  reset: () => void;
}

const IDLE: BackendStatus = {
  status: "idle",
  task: "",
  textinfo: null,
  current: "",
  step: 0,
  steps: 0,
  progress: 0,
  eta: null,
  elapsed: null,
  uptime: 0,
  connected: false,
  previewUrl: null,
};

export const useBackendStatusStore = create<BackendStatusState>()((set) => ({
  ...IDLE,

  setStatus: (data) =>
    set((state) => ({
      status: (data.status as string) ?? state.status,
      task: (data.task as string) ?? state.task,
      textinfo: (data.textinfo as string | null) ?? null,
      current: (data.current as string) ?? state.current,
      step: (data.step as number) ?? state.step,
      steps: (data.steps as number) ?? state.steps,
      progress: (data.progress as number) ?? state.progress,
      eta: (data.eta as number | null) ?? null,
      elapsed: (data.elapsed as number | null) ?? null,
      uptime: (data.uptime as number) ?? state.uptime,
    })),

  setPreview: (url) =>
    set((state) => {
      if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
      return { previewUrl: url };
    }),

  setConnected: (connected) => set({ connected }),

  reset: () =>
    set((state) => {
      if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
      return { ...IDLE, connected: state.connected };
    }),
}));
