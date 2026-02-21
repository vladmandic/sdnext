import { create } from "zustand";
import { persist } from "zustand/middleware";

interface VideoState {
  activeVideoTab: string;

  // Shared
  engine: string;
  model: string;
  prompt: string;
  negative: string;
  width: number;
  height: number;
  frames: number;
  steps: number;
  sampler: number;
  samplerShift: number;
  dynamicShift: boolean;
  seed: number;
  guidanceScale: number;
  guidanceTrue: number;
  initStrength: number;
  vaeType: string;
  vaeTileFrames: number;
  fps: number;
  interpolate: number;
  codec: string;
  format: string;
  codecOptions: string;
  saveVideo: boolean;
  saveFrames: boolean;
  saveSafetensors: boolean;

  // Shared input images (File objects, not persisted to localStorage)
  initImage: File | null;
  lastImage: File | null;

  // FramePack
  fpVariant: string;
  fpResolution: number;
  fpDuration: number;
  fpLatentWindowSize: number;
  fpSteps: number;
  fpShift: number;
  fpCfgScale: number;
  fpCfgDistilled: number;
  fpCfgRescale: number;
  fpStartWeight: number;
  fpEndWeight: number;
  fpVisionWeight: number;
  fpSectionPrompt: string;
  fpSystemPrompt: string;
  fpTeacache: boolean;
  fpOptimizedPrompt: boolean;
  fpCfgZero: boolean;
  fpPreview: boolean;
  fpAttention: string;
  fpVaeType: string;

  // LTX
  ltxModel: string;
  ltxSteps: number;
  ltxDecodeTimestep: number;
  ltxNoiseScale: number;
  ltxUpsampleEnable: boolean;
  ltxUpsampleRatio: number;
  ltxRefineEnable: boolean;
  ltxRefineStrength: number;
  ltxConditionStrength: number;
  ltxAudioEnable: boolean;

  // Runtime state
  resultVideoUrl: string | null;

  setParam: <K extends keyof VideoState>(key: K, value: VideoState[K]) => void;
  setParams: (params: Partial<VideoState>) => void;
  setResultVideo: (url: string | null) => void;
  reset: () => void;
}

const defaultParams = {
  activeVideoTab: "models",

  engine: "",
  model: "",
  prompt: "",
  negative: "",
  width: 848,
  height: 480,
  frames: 25,
  steps: 30,
  sampler: 0,
  samplerShift: -1,
  dynamicShift: false,
  seed: -1,
  guidanceScale: 6,
  guidanceTrue: -1,
  initStrength: 0.5,
  vaeType: "Default",
  vaeTileFrames: 0,
  fps: 24,
  interpolate: 0,
  codec: "libx264",
  format: "mp4",
  codecOptions: "crf:16",
  saveVideo: true,
  saveFrames: false,
  saveSafetensors: false,

  initImage: null as File | null,
  lastImage: null as File | null,

  fpVariant: "bi-directional",
  fpResolution: 640,
  fpDuration: 4,
  fpLatentWindowSize: 9,
  fpSteps: 25,
  fpShift: 3,
  fpCfgScale: 1,
  fpCfgDistilled: 10,
  fpCfgRescale: 0,
  fpStartWeight: 1,
  fpEndWeight: 1,
  fpVisionWeight: 1,
  fpSectionPrompt: "",
  fpSystemPrompt: "",
  fpTeacache: true,
  fpOptimizedPrompt: true,
  fpCfgZero: false,
  fpPreview: true,
  fpAttention: "Default",
  fpVaeType: "Full",

  ltxModel: "",
  ltxSteps: 50,
  ltxDecodeTimestep: 0.05,
  ltxNoiseScale: 0.025,
  ltxUpsampleEnable: false,
  ltxUpsampleRatio: 2,
  ltxRefineEnable: false,
  ltxRefineStrength: 0.4,
  ltxConditionStrength: 0.8,
  ltxAudioEnable: false,
};

const defaultParamKeys = Object.keys(defaultParams) as (keyof typeof defaultParams)[];

export const useVideoStore = create<VideoState>()(
  persist(
    (set) => ({
      ...defaultParams,

      resultVideoUrl: null,

      setParam: (key, value) => set({ [key]: value }),
      setParams: (params) => set(params),
      setResultVideo: (url) => set({ resultVideoUrl: url }),
      reset: () => set({ ...defaultParams }),
    }),
    {
      name: "sdnext-video",
      partialize: (state) => {
        const p: Record<string, unknown> = {};
        for (const key of defaultParamKeys) {
          if (key === "initImage" || key === "lastImage") continue;
          p[key] = state[key];
        }
        return p as Partial<VideoState>;
      },
    },
  ),
);
