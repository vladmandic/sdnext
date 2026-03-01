import type { ParsedGenerationInfo } from "@/api/types/gallery";

const RE_PARAM = /\s*([\w ]+):\s*("(?:\\"|[^"])*"|[^,]*)(?:,|$)/g;
const RE_SIZE = /^(\d+)x(\d+)$/;

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

function parseParams(text: string, params: Record<string, string>) {
  RE_PARAM.lastIndex = 0;
  let match: RegExpExecArray | null;
  while ((match = RE_PARAM.exec(text)) !== null) {
    const key = match[1].trim();
    let value = match[2].trim();
    if (value.startsWith('"') && value.endsWith('"')) {
      try { value = JSON.parse(value); } catch { value = value.slice(1, -1); }
    }
    if (!key || !value) continue;

    // Split "Size" → "Size-1" / "Size-2", "Batch" → batch count/size, etc.
    const sizeMatch = RE_SIZE.exec(value);
    if (sizeMatch) {
      if (key === "Size" || key === "Hires fixed" || key === "Hires size") {
        params[`${key}-1`] = sizeMatch[1];
        params[`${key}-2`] = sizeMatch[2];
      } else if (key === "Batch") {
        params["Batch count"] = sizeMatch[1];
        params["Batch size"] = sizeMatch[2];
      } else {
        params[key] = value;
      }
    } else {
      params[key] = value;
    }
  }
}
