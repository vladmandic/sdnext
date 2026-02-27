import type { FitMode, FreeTransform } from "@/lib/image";

export type ControlUnitType = "controlnet" | "t2i" | "xs" | "lite" | "reference" | "ip" | "asset";

export interface ControlUnit {
  enabled: boolean;
  unitType: ControlUnitType;
  processor: string;
  model: string;
  mode: string;
  strength: number;
  start: number;
  end: number;
  imageSource: string;
  image: File | null;
  imageDims: { w: number; h: number } | null;
  processedImage: string | null;
  // ControlNet-specific
  guess: boolean;
  // T2I-Adapter-specific
  factor: number;
  // Reference-specific
  attention: string;
  fidelity: number;
  queryWeight: number;
  adainWeight: number;
  // IP-Adapter-specific
  adapter: string;
  scale: number;
  crop: boolean;
  images: File[];
  masks: File[];
  fitMode: FitMode;
  freeTransform: FreeTransform | null;
  processorParams: Record<string, unknown>;
}

/** JSON-serializable snapshot of a ControlUnit for history persistence (File → base64 string). */
export interface ControlUnitSnapshot {
  enabled: boolean;
  unitType: ControlUnitType;
  imageSource: string;
  processor: string;
  model: string;
  mode: string;
  strength: number;
  start: number;
  end: number;
  image: string | null;
  imageDims: { w: number; h: number } | null;
  processedImage: string | null;
  guess: boolean;
  factor: number;
  attention: string;
  fidelity: number;
  queryWeight: number;
  adainWeight: number;
  adapter: string;
  scale: number;
  crop: boolean;
  images: string[];
  masks: string[];
  fitMode: FitMode;
  freeTransform: FreeTransform | null;
  processorParams: Record<string, unknown>;
}

export interface PreprocessorInfo {
  name: string;
  group: string;
  params: Record<string, unknown>;
}

export interface PreprocessResponse {
  ok: boolean;
  model: string;
  image: string;
}
