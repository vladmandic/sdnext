import { create } from "zustand";
import { putResult, trimResults, clearAllResults, getAllResults } from "@/lib/historyDb";
import type { MaskLine } from "@/stores/img2imgStore";
import type { ControlUnitSnapshot } from "@/api/types/control";

export interface GenerationResult {
  id: string;
  images: string[];
  parameters: Record<string, unknown>;
  info: string;
  timestamp: number;
  /** Flattened canvas base64 captured at generation time. Persisted to IndexedDB with the result. */
  inputImage?: string;
  /** Mask strokes captured at generation time. Persisted to IndexedDB with the result. */
  inputMask?: MaskLine[];
  /** Control unit settings + images captured at generation time. */
  controlUnits?: ControlUnitSnapshot[];
}

interface GenerationState {
  // Prompt
  prompt: string;
  negativePrompt: string;

  // Sampler
  sampler: string;
  steps: number;

  // Resolution
  width: number;
  height: number;

  // Batch
  batchSize: number;
  batchCount: number;

  // Guidance
  cfgScale: number;
  cfgEnd: number;
  guidanceRescale: number;
  imageCfgScale: number;
  pagScale: number;
  pagAdaptive: number;
  seed: number;
  subseed: number;
  subseedStrength: number;
  denoisingStrength: number;

  // Sampler / Scheduler
  sigmaMethod: string;
  timestepSpacing: string;
  betaSchedule: string;
  predictionMethod: string;
  timestepsPreset: string;
  timestepsOverride: string;
  sigmaAdjust: number;
  sigmaAdjustStart: number;
  sigmaAdjustEnd: number;
  flowShift: number;
  baseShift: number;
  maxShift: number;
  lowOrder: boolean;
  thresholding: boolean;
  dynamic: boolean;
  rescale: boolean;

  // Hires Fix
  hiresEnabled: boolean;
  hiresUpscaler: string;
  hiresScale: number;
  hiresSteps: number;
  hiresDenoising: number;
  hiresResizeMode: number;
  hiresSampler: string;
  hiresForce: boolean;
  hiresResizeX: number;
  hiresResizeY: number;
  hiresResizeContext: string;

  // Refiner
  refinerStart: number;
  refinerSteps: number;
  refinerPrompt: string;
  refinerNegative: string;

  // Advanced
  clipSkip: number;
  vaeType: string;
  tiling: boolean;
  hidiffusion: boolean;
  overrideSettings: Record<string, unknown>;

  // Detailer
  detailerEnabled: boolean;
  detailerModels: string[];
  detailerPrompt: string;
  detailerNegative: string;
  detailerSteps: number;
  detailerStrength: number;
  detailerResolution: number;
  detailerMaxDetected: number;
  detailerPadding: number;
  detailerBlur: number;
  detailerConfidence: number;
  detailerIou: number;
  detailerMinSize: number;
  detailerMaxSize: number;
  detailerRenoise: number;
  detailerRenoiseEnd: number;
  detailerSegmentation: boolean;
  detailerIncludeDetections: boolean;
  detailerMerge: boolean;
  detailerSort: boolean;
  detailerClasses: string;

  // Corrections
  hdrMode: number;
  hdrBrightness: number;
  hdrSharpen: number;
  hdrColor: number;
  hdrClamp: boolean;
  hdrBoundary: number;
  hdrThreshold: number;
  hdrMaximize: boolean;
  hdrMaxCenter: number;
  hdrMaxBoundary: number;
  hdrTintRatio: number;

  // Generation state
  isGenerating: boolean;
  currentTaskId: string | null;
  progress: number;
  eta: number;
  previewImage: string | null;
  queuePosition: number;

  // Results
  results: GenerationResult[];
  selectedResultId: string | null;
  selectedImageIndex: number | null;
  _historyLimit: number;

  // Actions
  setParam: <K extends keyof GenerationState>(key: K, value: GenerationState[K]) => void;
  setParams: (params: Partial<GenerationState>) => void;
  setGenerating: (generating: boolean) => void;
  setProgress: (progress: number, eta: number) => void;
  setPreview: (image: string | null) => void;
  setTaskId: (id: string | null) => void;
  addResult: (result: GenerationResult) => void;
  clearResults: () => void;
  selectImage: (resultId: string, imageIndex: number) => void;
  clearSelection: () => void;
  setHistoryLimit: (limit: number) => void;
  hydrateFromDb: () => void;
  reset: () => void;
}

const defaultParams = {
  prompt: "",
  negativePrompt: "",
  sampler: "Euler",
  steps: 20,
  width: 1024,
  height: 1024,
  batchSize: 1,
  batchCount: 1,
  cfgScale: 7,
  cfgEnd: 1,
  guidanceRescale: 0,
  imageCfgScale: 6,
  pagScale: 0,
  pagAdaptive: 0.5,
  seed: -1,
  subseed: -1,
  subseedStrength: 0,
  denoisingStrength: 0.5,
  sigmaMethod: "default",
  timestepSpacing: "default",
  betaSchedule: "default",
  predictionMethod: "default",
  timestepsPreset: "None",
  timestepsOverride: "",
  sigmaAdjust: 1.0,
  sigmaAdjustStart: 0.2,
  sigmaAdjustEnd: 1.0,
  flowShift: 3,
  baseShift: 0.5,
  maxShift: 1.15,
  lowOrder: true,
  thresholding: false,
  dynamic: false,
  rescale: false,
  hiresEnabled: false,
  hiresUpscaler: "Latent",
  hiresScale: 2,
  hiresSteps: 0,
  hiresDenoising: 0.5,
  hiresResizeMode: 0,
  hiresSampler: "",
  hiresForce: false,
  hiresResizeX: 0,
  hiresResizeY: 0,
  hiresResizeContext: "None",
  refinerStart: 0,
  refinerSteps: 0,
  refinerPrompt: "",
  refinerNegative: "",
  clipSkip: 1,
  vaeType: "Full",
  tiling: false,
  hidiffusion: false,
  overrideSettings: {},
  detailerEnabled: false,
  detailerModels: ["face-yolo8n"],
  detailerPrompt: "",
  detailerNegative: "",
  detailerSteps: 10,
  detailerStrength: 0.3,
  detailerResolution: 1024,
  detailerMaxDetected: 2,
  detailerPadding: 20,
  detailerBlur: 10,
  detailerConfidence: 0.6,
  detailerIou: 0.5,
  detailerMinSize: 0.0,
  detailerMaxSize: 1.0,
  detailerRenoise: 1.0,
  detailerRenoiseEnd: 1.0,
  detailerSegmentation: false,
  detailerIncludeDetections: false,
  detailerMerge: false,
  detailerSort: false,
  detailerClasses: "",
  hdrMode: 0,
  hdrBrightness: 0,
  hdrSharpen: 0,
  hdrColor: 0,
  hdrClamp: false,
  hdrBoundary: 4.0,
  hdrThreshold: 0.95,
  hdrMaximize: false,
  hdrMaxCenter: 0.6,
  hdrMaxBoundary: 1.0,
  hdrTintRatio: 0,
};

export const useGenerationStore = create<GenerationState>()((set) => ({
  ...defaultParams,

  isGenerating: false,
  currentTaskId: null,
  progress: 0,
  eta: 0,
  previewImage: null,
  queuePosition: 0,
  results: [],
  selectedResultId: null,
  selectedImageIndex: null,
  _historyLimit: 16,

  setParam: (key, value) => set({ [key]: value }),

  setParams: (params) => set(params),

  setGenerating: (generating) =>
    set({
      isGenerating: generating,
      ...(generating ? {} : { progress: 0, eta: 0, previewImage: null }),
    }),

  setProgress: (progress, eta) => set({ progress, eta }),

  setPreview: (image) => set({ previewImage: image }),

  setTaskId: (id) => set({ currentTaskId: id }),

  addResult: (result) =>
    set((state) => {
      putResult(result).then(() => trimResults(state._historyLimit));
      return {
        results: [result, ...state.results].slice(0, 100),
        selectedResultId: result.id,
        selectedImageIndex: 0,
      };
    }),

  clearResults: () => {
    clearAllResults();
    set({ results: [], selectedResultId: null, selectedImageIndex: null });
  },

  selectImage: (resultId, imageIndex) =>
    set({ selectedResultId: resultId, selectedImageIndex: imageIndex }),

  clearSelection: () =>
    set({ selectedResultId: null, selectedImageIndex: null }),

  setHistoryLimit: (limit) => set({ _historyLimit: limit }),

  hydrateFromDb: () => {
    getAllResults().then((dbResults) => {
      if (useGenerationStore.getState().results.length === 0 && dbResults.length > 0) {
        useGenerationStore.setState({
          results: dbResults,
          selectedResultId: dbResults[0]?.id ?? null,
          selectedImageIndex: dbResults[0] ? 0 : null,
        });
      }
    });
  },

  reset: () => set({ ...defaultParams }),
}));
