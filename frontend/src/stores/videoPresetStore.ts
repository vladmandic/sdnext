import { create } from "zustand";
import { persist } from "zustand/middleware";
import { useVideoStore } from "@/stores/videoStore";

export type PresetDomain = "video" | "framepack" | "ltx";

export interface VideoPreset {
  id: string;
  name: string;
  domain: PresetDomain;
  params: Record<string, unknown>;
  createdAt: number;
  builtIn?: boolean;
}

const SHARED_PARAM_KEYS = ["width", "height", "frames", "steps", "sampler", "samplerShift", "dynamicShift", "seed", "guidanceScale", "guidanceTrue", "initStrength", "vaeType", "vaeTileFrames"] as const;

const OUTPUT_PARAM_KEYS = ["fps", "interpolate", "codec", "format", "codecOptions", "outputPreset", "outputQuality", "saveVideo", "saveFrames", "saveSafetensors"] as const;

const FP_PARAM_KEYS = ["fpVariant", "fpResolution", "fpDuration", "fpLatentWindowSize", "fpSteps", "fpShift", "fpCfgScale", "fpCfgDistilled", "fpCfgRescale", "fpStartWeight", "fpEndWeight", "fpVisionWeight", "fpSectionPrompt", "fpSystemPrompt", "fpTeacache", "fpOptimizedPrompt", "fpCfgZero", "fpPreview", "fpAttention", "fpVaeType"] as const;

const LTX_PARAM_KEYS = ["ltxModel", "ltxSteps", "ltxDecodeTimestep", "ltxNoiseScale", "ltxUpsampleEnable", "ltxUpsampleRatio", "ltxRefineEnable", "ltxRefineStrength", "ltxConditionStrength", "ltxAudioEnable"] as const;

function keysForDomain(domain: PresetDomain): readonly string[] {
  const base: string[] = [...SHARED_PARAM_KEYS, ...OUTPUT_PARAM_KEYS];
  if (domain === "framepack") return [...base, ...FP_PARAM_KEYS];
  if (domain === "ltx") return [...base, ...LTX_PARAM_KEYS];
  return base;
}

export function snapshotParams(domain: PresetDomain): Record<string, unknown> {
  const state = useVideoStore.getState() as unknown as Record<string, unknown>;
  const keys = keysForDomain(domain);
  const snap: Record<string, unknown> = {};
  for (const k of keys) {
    snap[k] = state[k];
  }
  return snap;
}

function makeId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

const BUILT_IN_PRESETS: VideoPreset[] = [
  {
    id: "builtin-fp-fast",
    name: "FramePack Fast",
    domain: "framepack",
    builtIn: true,
    createdAt: 0,
    params: { fpResolution: 480, fpSteps: 15, fpTeacache: true, fpDuration: 4, fpLatentWindowSize: 9 },
  },
  {
    id: "builtin-fp-quality",
    name: "FramePack Quality",
    domain: "framepack",
    builtIn: true,
    createdAt: 0,
    params: { fpResolution: 832, fpSteps: 30, fpTeacache: false, fpDuration: 4, fpLatentWindowSize: 9 },
  },
  {
    id: "builtin-ltx-quick",
    name: "LTX Quick",
    domain: "ltx",
    builtIn: true,
    createdAt: 0,
    params: { ltxSteps: 25, ltxDecodeTimestep: 0.1, ltxUpsampleEnable: false, ltxRefineEnable: false },
  },
  {
    id: "builtin-ltx-full",
    name: "LTX Full",
    domain: "ltx",
    builtIn: true,
    createdAt: 0,
    params: { ltxSteps: 50, ltxUpsampleEnable: true, ltxUpsampleRatio: 2, ltxRefineEnable: true, ltxRefineStrength: 0.4 },
  },
];

interface VideoPresetState {
  presets: VideoPreset[];
  addPreset: (preset: Omit<VideoPreset, "id" | "createdAt">) => void;
  removePreset: (id: string) => void;
  renamePreset: (id: string, name: string) => void;
  importPresets: (json: string) => number;
  exportPresets: (domain?: PresetDomain) => string;
  getPresetsForDomain: (domain: PresetDomain) => VideoPreset[];
}

export const useVideoPresetStore = create<VideoPresetState>()(
  persist(
    (set, get) => ({
      presets: [],

      addPreset: (preset) => {
        const newPreset: VideoPreset = {
          ...preset,
          id: makeId(),
          createdAt: Date.now(),
        };
        set((s) => ({ presets: [...s.presets, newPreset] }));
      },

      removePreset: (id) => {
        set((s) => ({ presets: s.presets.filter((p) => p.id !== id) }));
      },

      renamePreset: (id, name) => {
        set((s) => ({
          presets: s.presets.map((p) => (p.id === id ? { ...p, name } : p)),
        }));
      },

      importPresets: (json) => {
        try {
          const parsed = JSON.parse(json);
          const arr = Array.isArray(parsed) ? parsed : [];
          const valid = arr.filter(
            (p: unknown): p is VideoPreset =>
              typeof p === "object" && p !== null && "name" in p && "domain" in p && "params" in p,
          );
          const imported = valid.map((p) => ({
            ...p,
            id: makeId(),
            createdAt: Date.now(),
            builtIn: false,
          }));
          set((s) => ({ presets: [...s.presets, ...imported] }));
          return imported.length;
        } catch {
          return 0;
        }
      },

      exportPresets: (domain) => {
        const all = get().presets.filter((p) => !p.builtIn);
        const filtered = domain ? all.filter((p) => p.domain === domain) : all;
        return JSON.stringify(
          filtered.map(({ id: _id, createdAt: _ts, ...rest }) => rest),
          null,
          2,
        );
      },

      getPresetsForDomain: (domain) => {
        const builtIn = BUILT_IN_PRESETS.filter((p) => p.domain === domain);
        const user = get().presets.filter((p) => p.domain === domain);
        return [...builtIn, ...user];
      },
    }),
    {
      name: "sdnext-video-presets",
      partialize: (state) => ({ presets: state.presets.filter((p) => !p.builtIn) }),
    },
  ),
);
