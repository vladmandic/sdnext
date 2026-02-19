import { useGenerationStore } from "@/stores/generationStore";

const KEY_MAP: Record<string, string> = {
  Prompt: "prompt",
  "Negative prompt": "negativePrompt",
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
};

const NUM_KEYS = new Set(["steps", "cfgScale", "seed", "width", "height", "denoisingStrength", "clipSkip", "batchSize", "batchCount"]);

export function restoreFromPngInfo(parameters: Record<string, unknown>) {
  const update: Record<string, unknown> = {};
  for (const [pngKey, storeKey] of Object.entries(KEY_MAP)) {
    const val = parameters[pngKey];
    if (val === undefined || val === null) continue;
    if (NUM_KEYS.has(storeKey)) {
      const n = Number(val);
      if (!Number.isNaN(n)) update[storeKey] = n;
    } else {
      update[storeKey] = String(val);
    }
  }
  if (Object.keys(update).length > 0) {
    useGenerationStore.getState().setParams(update);
  }
}
