export type ControlUnitType = "controlnet" | "t2i" | "xs" | "lite" | "reference";

export interface ControlUnit {
  enabled: boolean;
  unitType: ControlUnitType;
  processor: string;
  model: string;
  strength: number;
  start: number;
  end: number;
  image: File | null;
}

export interface PreprocessorInfo {
  name: string;
  params: Record<string, unknown>;
}
