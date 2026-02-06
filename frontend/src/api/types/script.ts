export interface ScriptArg {
  label: string;
  value: unknown;
  minimum?: number;
  maximum?: number;
  step?: number;
  choices?: string[];
}

export interface ScriptInfo {
  name: string;
  is_alwayson: boolean;
  is_img2img: boolean;
  args: ScriptArg[];
}

export interface ScriptsList {
  txt2img: string[];
  img2img: string[];
  control: string[];
}
