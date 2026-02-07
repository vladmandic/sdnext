import type { ParsedGenerationInfo } from "@/api/types/gallery";

export function parseGenerationInfo(raw: string | null | undefined): ParsedGenerationInfo {
  const result: ParsedGenerationInfo = { prompt: "", negativePrompt: "", params: {} };
  if (!raw) return result;

  const negIdx = raw.indexOf("Negative prompt:");
  const stepsIdx = raw.search(/\nSteps:\s/);

  if (negIdx === -1 && stepsIdx === -1) {
    result.prompt = raw.trim();
    return result;
  }

  if (negIdx >= 0) {
    result.prompt = raw.slice(0, negIdx).trim();
    const afterNeg = raw.slice(negIdx + "Negative prompt:".length);
    const paramsStart = afterNeg.search(/\nSteps:\s/);
    if (paramsStart >= 0) {
      result.negativePrompt = afterNeg.slice(0, paramsStart).trim();
      parseParams(afterNeg.slice(paramsStart).trim(), result.params);
    } else {
      result.negativePrompt = afterNeg.trim();
    }
  } else if (stepsIdx >= 0) {
    result.prompt = raw.slice(0, stepsIdx).trim();
    parseParams(raw.slice(stepsIdx).trim(), result.params);
  }

  return result;
}

function parseParams(line: string, params: Record<string, string>) {
  const pairs = line.split(",");
  for (const pair of pairs) {
    const colonIdx = pair.indexOf(":");
    if (colonIdx > 0) {
      const key = pair.slice(0, colonIdx).trim();
      const value = pair.slice(colonIdx + 1).trim();
      if (key && value) params[key] = value;
    }
  }
}
