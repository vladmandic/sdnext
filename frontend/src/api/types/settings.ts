export type OptionsMap = Record<string, unknown>;

export interface CmdFlags {
  [key: string]: unknown;
}

export interface OptionInfoMeta {
  label: string;
  section_id: string | null;
  section_title: string;
  visible: boolean;
  hidden: boolean;
  type: "boolean" | "number" | "string" | "array";
  component: "slider" | "switch" | "radio" | "dropdown" | "input" | "number" | "color" | "checkboxgroup" | "separator";
  component_args: {
    minimum?: number;
    maximum?: number;
    step?: number;
    choices?: string[];
    precision?: number;
    multiselect?: boolean;
  };
  is_legacy: boolean;
}

export interface SectionMeta {
  id: string;
  title: string;
  hidden: boolean;
}

export interface OptionsInfoResponse {
  options: Record<string, OptionInfoMeta>;
  sections: SectionMeta[];
}
