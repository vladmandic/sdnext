import { useGenerationStore } from "@/stores/generationStore";

const KEY_MAP: Record<string, string> = {
  // Prompt
  Prompt: "prompt",
  "Negative prompt": "negativePrompt",

  // Basic
  Steps: "steps",
  "CFG scale": "cfgScale",
  Seed: "seed",
  "Size-1": "width",
  "Size-2": "height",
  Sampler: "sampler",
  "Denoising strength": "denoisingStrength",
  "Clip skip": "clipSkip",
  "Batch size": "batchSize",
  "Batch count": "batchCount",

  // Guidance
  "CFG end": "cfgEnd",
  "CFG rescale": "guidanceRescale",
  "Image CFG scale": "imageCfgScale",
  "CFG true": "pagScale",
  "CFG adaptive": "pagAdaptive",
  "Variation seed": "subseed",
  "Variation strength": "subseedStrength",
  Tiling: "tiling",

  // Hires
  "Hires upscaler": "hiresUpscaler",
  "Hires scale": "hiresScale",
  "Hires steps": "hiresSteps",
  "Hires strength": "hiresDenoising",
  "Hires sampler": "hiresSampler",
  "Hires fixed-1": "hiresResizeX",
  "Hires fixed-2": "hiresResizeY",
  "HiRes mode": "hiresResizeMode",
  "Hires force": "hiresForce",
  "HiRes context": "hiresResizeContext",

  // Refiner
  "Refiner start": "refinerStart",
  "Refiner steps": "refinerSteps",
  "Refiner prompt": "refinerPrompt",
  "Refiner negative": "refinerNegative",

  // VAE
  "VAE type": "vaeType",

  // Scheduler
  "Sampler sigma": "sigmaMethod",
  "Sampler spacing": "timestepSpacing",
  "Sampler beta schedule": "betaSchedule",
  "Sampler type": "predictionMethod",
  "Sampler shift": "flowShift",
  "Sampler low order": "lowOrder",
  "Sampler dynamic": "thresholding",
  "Sampler rescale": "rescale",

  // Token merging
  ToMe: "tomeRatio",
  ToDo: "todoRatio",
};

const NUM_KEYS = new Set([
  "steps", "cfgScale", "seed", "width", "height", "denoisingStrength", "clipSkip",
  "batchSize", "batchCount", "cfgEnd", "guidanceRescale", "imageCfgScale",
  "pagScale", "pagAdaptive", "subseed", "subseedStrength",
  "hiresScale", "hiresSteps", "hiresDenoising", "hiresResizeX", "hiresResizeY", "hiresResizeMode",
  "refinerStart", "refinerSteps",
  "flowShift", "tomeRatio", "todoRatio",
]);

const BOOL_KEYS = new Set([
  "tiling", "hiresForce", "lowOrder", "thresholding", "rescale",
]);

export function restoreFromPngInfo(parameters: Record<string, unknown>) {
  const update: Record<string, unknown> = {};
  for (const [pngKey, storeKey] of Object.entries(KEY_MAP)) {
    const val = parameters[pngKey];
    if (val === undefined || val === null) continue;
    if (NUM_KEYS.has(storeKey)) {
      const n = Number(val);
      if (!Number.isNaN(n)) update[storeKey] = n;
    } else if (BOOL_KEYS.has(storeKey)) {
      update[storeKey] = val === true || val === "True" || val === "true" || val === "1";
    } else {
      update[storeKey] = String(val);
    }
  }

  // Auto-enable hires when any hires param is present
  if (update.hiresUpscaler || update.hiresScale || update.hiresSteps || update.hiresDenoising || update.hiresResizeX || update.hiresResizeY) {
    update.hiresEnabled = true;
  }

  if (Object.keys(update).length > 0) {
    useGenerationStore.getState().setParams(update);
  }
}
