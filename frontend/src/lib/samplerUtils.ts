import type { Sampler } from "@/api/types/models";

export type SamplerGroup = "Standard" | "FlowMatch" | "Res4Lyf";

const RES4LYF_KEYS = ["shift", "base_shift", "max_shift"] as const;

const FLOW_MODEL_TYPES = new Set([
  "f1", "f2", "sd3", "lumina", "auraflow", "sana", "hidream",
  "flux", "stable diffusion 3", "hunyuan",
]);

export function classifySampler(s: Sampler): SamplerGroup {
  if (s.name.includes("FlowMatch")) return "FlowMatch";
  const keys = Object.keys(s.options);
  if (RES4LYF_KEYS.every((k) => keys.includes(k))) return "Res4Lyf";
  return "Standard";
}

export function isSamplerCompatible(group: SamplerGroup, modelType: string | null | undefined): boolean {
  if (!modelType) return true;
  const lower = modelType.toLowerCase();
  const isFlow = FLOW_MODEL_TYPES.has(lower) || lower.includes("flux") || lower.includes("flow");
  const isSDXL = lower.includes("sdxl") || lower.includes("xl");
  if (isSDXL) return true;
  if (isFlow) return group === "FlowMatch" || group === "Res4Lyf";
  return group === "Standard";
}
