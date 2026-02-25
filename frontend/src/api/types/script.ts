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

export interface ScriptInfoV2 {
  name: string;
  is_alwayson: boolean;
  contexts: string[];
  args: ScriptArg[];
}

export interface ScriptsResponse {
  scripts: ScriptInfoV2[];
}
