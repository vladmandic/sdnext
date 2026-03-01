import type { GenerationState } from "@/stores/generationStore";

type PartialGenParams = Partial<Pick<GenerationState,
  "width" | "height" | "cfgScale" | "steps" | "sampler" | "flowShift" | "denoisingStrength"
>>;

const MODEL_DEFAULTS: Record<string, PartialGenParams> = {
  sd:   { width: 512, height: 512, cfgScale: 7, steps: 20, sampler: "DPM++ 2M" },
  sdxl: { width: 1024, height: 1024, cfgScale: 7, steps: 20, sampler: "DPM++ 2M" },
  sd3:  { width: 1024, height: 1024, cfgScale: 5, steps: 28, sampler: "Euler" },
  f1:   { width: 1024, height: 1024, cfgScale: 1, steps: 20, sampler: "Euler", flowShift: 3 },
  f2:   { width: 1024, height: 1024, cfgScale: 1, steps: 20, sampler: "Euler", flowShift: 3 },
  sc:   { width: 1024, height: 1024, cfgScale: 4, steps: 20 },
};

export function getModelDefaults(type: string | null | undefined): PartialGenParams | null {
  if (!type) return null;
  return MODEL_DEFAULTS[type] ?? null;
}

export function formatSuggestion(defaults: PartialGenParams): string {
  const parts: string[] = [];
  if (defaults.width && defaults.height) parts.push(`${defaults.width}x${defaults.height}`);
  if (defaults.sampler) parts.push(defaults.sampler);
  if (defaults.steps) parts.push(`${defaults.steps} steps`);
  if (defaults.cfgScale !== undefined) parts.push(`CFG ${defaults.cfgScale}`);
  if (defaults.flowShift !== undefined) parts.push(`shift ${defaults.flowShift}`);
  return parts.join(", ");
}
