export type ControlUnitType = "controlnet" | "t2i" | "xs" | "lite" | "reference" | "ip" | "asset";

export interface ControlUnit {
  enabled: boolean;
  unitType: ControlUnitType;
  processor: string;
  model: string;
  strength: number;
  start: number;
  end: number;
  image: File | null;
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
}

export interface PreprocessorInfo {
  name: string;
  params: Record<string, unknown>;
}
