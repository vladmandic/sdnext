import { create } from "zustand";
import { persist } from "zustand/middleware";
import { useGenerationStore } from "@/stores/generationStore";

export interface GenerationPreset {
  id: string;
  name: string;
  compatibleTypes: string[];
  params: Record<string, unknown>;
  createdAt: number;
  builtIn?: boolean;
}

const PRESET_PARAM_KEYS = [
  "sampler", "steps", "width", "height", "cfgScale", "cfgEnd", "denoisingStrength",
  "clipSkip", "vaeType", "flowShift", "baseShift", "maxShift",
  "sigmaMethod", "timestepSpacing", "betaSchedule", "predictionMethod",
  "hiresEnabled", "hiresUpscaler", "hiresScale", "hiresSteps", "hiresDenoising",
  "hiresResizeMode", "hiresSampler",
] as const;

export function snapshotParams(): Record<string, unknown> {
  const state = useGenerationStore.getState() as unknown as Record<string, unknown>;
  const snap: Record<string, unknown> = {};
  for (const k of PRESET_PARAM_KEYS) {
    snap[k] = state[k];
  }
  return snap;
}

function makeId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

const ALL_TYPES = ["sd", "sdxl", "sd3", "f1", "f2", "sc", "auraflow", "hunyuandit", "pixart", "kandinsky"];

const BUILT_IN_PRESETS: GenerationPreset[] = [
  {
    id: "builtin-sdxl-default",
    name: "SDXL Default",
    compatibleTypes: ["sdxl"],
    builtIn: true,
    createdAt: 0,
    params: { width: 1024, height: 1024, sampler: "DPM++ 2M", steps: 20, cfgScale: 7 },
  },
  {
    id: "builtin-sd15-default",
    name: "SD 1.5 Default",
    compatibleTypes: ["sd"],
    builtIn: true,
    createdAt: 0,
    params: { width: 512, height: 512, sampler: "DPM++ 2M", steps: 20, cfgScale: 7 },
  },
  {
    id: "builtin-flux-default",
    name: "Flux Default",
    compatibleTypes: ["f1", "f2"],
    builtIn: true,
    createdAt: 0,
    params: { width: 1024, height: 1024, sampler: "Euler", steps: 20, cfgScale: 1, flowShift: 3 },
  },
  {
    id: "builtin-sd3-default",
    name: "SD3 Default",
    compatibleTypes: ["sd3"],
    builtIn: true,
    createdAt: 0,
    params: { width: 1024, height: 1024, sampler: "Euler", steps: 28, cfgScale: 5 },
  },
  {
    id: "builtin-portrait",
    name: "Portrait",
    compatibleTypes: ALL_TYPES,
    builtIn: true,
    createdAt: 0,
    params: { width: 768, height: 1152, hiresEnabled: true, hiresScale: 1.5 },
  },
  {
    id: "builtin-landscape",
    name: "Landscape",
    compatibleTypes: ALL_TYPES,
    builtIn: true,
    createdAt: 0,
    params: { width: 1344, height: 768 },
  },
  {
    id: "builtin-anime",
    name: "Anime",
    compatibleTypes: ["sd", "sdxl"],
    builtIn: true,
    createdAt: 0,
    params: { sampler: "Euler a", steps: 28, cfgScale: 8, clipSkip: 2 },
  },
  {
    id: "builtin-photo-real",
    name: "Photo-real",
    compatibleTypes: ["sd", "sdxl"],
    builtIn: true,
    createdAt: 0,
    params: { sampler: "DPM++ 3M SDE", steps: 30, cfgScale: 4.5, hiresEnabled: true, hiresScale: 2 },
  },
  {
    id: "builtin-fast-draft",
    name: "Fast Draft",
    compatibleTypes: ALL_TYPES,
    builtIn: true,
    createdAt: 0,
    params: { sampler: "Euler", steps: 4, cfgScale: 1 },
  },
];

interface PresetState {
  presets: GenerationPreset[];
  addPreset: (preset: Omit<GenerationPreset, "id" | "createdAt">) => void;
  removePreset: (id: string) => void;
  renamePreset: (id: string, name: string) => void;
  importPresets: (json: string) => number;
  exportPresets: () => string;
  getPresetsForModelType: (modelType: string | null) => GenerationPreset[];
}

export const usePresetStore = create<PresetState>()(
  persist(
    (set, get) => ({
      presets: [],

      addPreset: (preset) => {
        const newPreset: GenerationPreset = {
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
            (p: unknown): p is GenerationPreset =>
              typeof p === "object" && p !== null && "name" in p && "params" in p,
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

      exportPresets: () => {
        const all = get().presets.filter((p) => !p.builtIn);
        return JSON.stringify(
          all.map(({ id: _id, createdAt: _ts, ...rest }) => rest),
          null,
          2,
        );
      },

      getPresetsForModelType: (modelType) => {
        const builtIn = modelType
          ? BUILT_IN_PRESETS.filter((p) => p.compatibleTypes.includes(modelType))
          : BUILT_IN_PRESETS;
        const user = get().presets;
        return [...builtIn, ...user];
      },
    }),
    {
      name: "sdnext-generation-presets",
      partialize: (state) => ({ presets: state.presets.filter((p) => !p.builtIn) }),
    },
  ),
);
