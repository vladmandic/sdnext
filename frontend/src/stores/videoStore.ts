import { create } from "zustand";
import { persist } from "zustand/middleware";

interface VideoState {
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

  isGenerating: boolean;
  jobId: string | null;
  progress: number;
  resultVideoUrl: string | null;

  setParam: <K extends keyof VideoState>(key: K, value: VideoState[K]) => void;
  setParams: (params: Partial<VideoState>) => void;
  setGenerating: (generating: boolean) => void;
  setJobId: (id: string | null) => void;
  setProgress: (progress: number) => void;
  setResultVideo: (url: string | null) => void;
  reset: () => void;
}

const defaultParams = {
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
};

const defaultParamKeys = Object.keys(defaultParams) as (keyof typeof defaultParams)[];

export const useVideoStore = create<VideoState>()(
  persist(
    (set) => ({
      ...defaultParams,

      isGenerating: false,
      jobId: null,
      progress: 0,
      resultVideoUrl: null,

      setParam: (key, value) => set({ [key]: value }),
      setParams: (params) => set(params),
      setGenerating: (generating) => set({ isGenerating: generating, ...(generating ? {} : { progress: 0 }) }),
      setJobId: (id) => set({ jobId: id }),
      setProgress: (progress) => set({ progress }),
      setResultVideo: (url) => set({ resultVideoUrl: url }),
      reset: () => set({ ...defaultParams }),
    }),
    {
      name: "sdnext-video",
      partialize: (state) => {
        const p: Record<string, unknown> = {};
        for (const key of defaultParamKeys) p[key] = state[key];
        return p as Partial<VideoState>;
      },
    },
  ),
);
