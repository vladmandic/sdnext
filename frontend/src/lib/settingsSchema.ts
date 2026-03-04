import type { OptionInfoMeta } from "@/api/types/settings";
import { getParamHelpPlain } from "@/data/parameterHelp";

export type SettingComponent = "slider" | "switch" | "select" | "radio" | "input" | "number" | "color" | "separator" | "multiselect" | "path";

export interface SettingDef {
  key: string;
  label: string;
  component: SettingComponent;
  defaultValue?: unknown;
  min?: number;
  max?: number;
  step?: number;
  precision?: number;
  choices?: string[];
  description?: string;
  requiresRestart?: boolean;
  isSecret?: boolean;
  baseFolderKey?: string;
}

export interface SettingSectionDef {
  id: string;
  title: string;
  settings: SettingDef[];
}

export const settingsSchema: SettingSectionDef[] = [
  {
    id: "sd",
    title: "Model Loading",
    settings: [
      { key: "sd_model_checkpoint", label: "Checkpoint", component: "select" },
      { key: "sd_model_refiner", label: "Refiner", component: "select", defaultValue: "None" },
      { key: "sd_checkpoint_autoload", label: "Auto-load model", component: "switch", defaultValue: true },
      { key: "diffusers_pipeline", label: "Pipeline", component: "select" },
    ],
  },
  {
    id: "offload",
    title: "Model Offloading",
    settings: [
      { key: "diffusers_offload_mode", label: "Offload mode", component: "select", defaultValue: "none" },
    ],
  },
  {
    id: "vae_encoder",
    title: "VAE",
    settings: [
      { key: "sd_vae", label: "VAE model", component: "select", defaultValue: "Automatic" },
      { key: "no_half_vae", label: "Full precision VAE", component: "switch", defaultValue: false },
      { key: "diffusers_vae_upcast", label: "VAE upcast", component: "select", defaultValue: "default" },
      { key: "diffusers_vae_slicing", label: "VAE slicing", component: "switch", defaultValue: false },
      { key: "diffusers_vae_tiling", label: "VAE tiling", component: "switch", defaultValue: false },
    ],
  },
  {
    id: "text_encoder",
    title: "Text Encoder",
    settings: [
      { key: "sd_text_encoder", label: "Text encoder model", component: "select" },
      { key: "prompt_attention", label: "Prompt attention", component: "select" },
      { key: "prompt_mean_norm", label: "Mean normalization", component: "switch", defaultValue: true },
      { key: "comma_padding_backtrack", label: "Comma padding", component: "number", min: 0, max: 74, defaultValue: 20 },
    ],
  },
  {
    id: "cuda",
    title: "Compute",
    settings: [
      { key: "precision", label: "Precision", component: "select", defaultValue: "Autocast" },
      { key: "cuda_dtype", label: "CUDA dtype", component: "select" },
      { key: "no_half", label: "Disable half precision", component: "switch", defaultValue: false },
      { key: "cross_attention_optimization", label: "Cross attention", component: "select", defaultValue: "Automatic" },
      { key: "cuda_compile_backend", label: "Compile backend", component: "select", defaultValue: "none" },
    ],
  },
  {
    id: "saving-images",
    title: "Image Options",
    settings: [
      { key: "samples_save", label: "Save images", component: "switch", defaultValue: true },
      { key: "samples_format", label: "Image format", component: "select", defaultValue: "jpg" },
      { key: "samples_filename_pattern", label: "Filename pattern", component: "input", defaultValue: "" },
      { key: "jpeg_quality", label: "JPEG quality", component: "slider", min: 1, max: 100, step: 1, defaultValue: 85 },
      { key: "save_images_add_number", label: "Add number", component: "switch", defaultValue: true },
      { key: "grid_save", label: "Save grids", component: "switch", defaultValue: true },
      { key: "grid_format", label: "Grid format", component: "select", defaultValue: "jpg" },
      { key: "use_save_to_dirs_for_ui", label: "Save to subdirs", component: "switch", defaultValue: false },
    ],
  },
  {
    id: "saving-paths",
    title: "Image Paths",
    settings: [
      { key: "outdir_samples", label: "Output folder", component: "input" },
      { key: "outdir_txt2img_samples", label: "Text-to-image", component: "path", baseFolderKey: "outdir_samples" },
      { key: "outdir_img2img_samples", label: "Image-to-image", component: "path", baseFolderKey: "outdir_samples" },
      { key: "outdir_control_samples", label: "Control", component: "path", baseFolderKey: "outdir_samples" },
      { key: "outdir_extras_samples", label: "Extras", component: "path", baseFolderKey: "outdir_samples" },
      { key: "outdir_save", label: "Manual save", component: "path", baseFolderKey: "outdir_samples" },
      { key: "outdir_video", label: "Video", component: "path", baseFolderKey: "outdir_samples" },
      { key: "outdir_init_images", label: "Init images", component: "path", baseFolderKey: "outdir_samples" },
      { key: "outdir_grids", label: "Grids folder", component: "input" },
      { key: "outdir_txt2img_grids", label: "Text-to-image grids", component: "path", baseFolderKey: "outdir_grids" },
      { key: "outdir_img2img_grids", label: "Image-to-image grids", component: "path", baseFolderKey: "outdir_grids" },
      { key: "outdir_control_grids", label: "Control grids", component: "path", baseFolderKey: "outdir_grids" },
    ],
  },
  {
    id: "live-preview",
    title: "Live Previews",
    settings: [
      { key: "show_progress_every_n_steps", label: "Preview interval", component: "number", min: 0, max: 50, step: 1, defaultValue: 1 },
      { key: "show_progress_type", label: "Preview method", component: "select", defaultValue: "TAESD" },
      { key: "live_previews_enable", label: "Enable previews", component: "switch", defaultValue: true },
    ],
  },
  {
    id: "postprocessing",
    title: "Postprocessing",
    settings: [
      { key: "postprocessing_enable_in_main_ui", label: "Enable in main UI", component: "switch", defaultValue: false },
      { key: "upscaler_for_img2img", label: "Img2img upscaler", component: "select" },
      { key: "face_restoration", label: "Face restoration", component: "switch", defaultValue: false },
      { key: "face_restoration_model", label: "Face model", component: "select", defaultValue: "CodeFormer" },
      { key: "code_former_weight", label: "CodeFormer weight", component: "slider", min: 0, max: 1, step: 0.05, defaultValue: 0.2 },
    ],
  },
];

export const QUICK_SETTINGS_GROUPS = [
  { title: "Model", keys: ["sd_model_checkpoint", "sd_vae", "sd_model_refiner", "sd_model_dict"] },
  { title: "Performance", keys: ["cross_attention_optimization", "cuda_compile_backend", "cuda_compile_mode", "diffusers_offload_mode"] },
  { title: "Output", keys: ["samples_save", "save_init_img", "image_watermark_enabled", "batch_frame_mode"] },
  { title: "Preview", keys: ["live_previews_enable"] },
];

export function getSettingsMap(): Map<string, { section: SettingSectionDef; setting: SettingDef }> {
  const map = new Map<string, { section: SettingSectionDef; setting: SettingDef }>();
  for (const section of settingsSchema) {
    for (const setting of section.settings) {
      map.set(setting.key, { section, setting });
    }
  }
  return map;
}

const metaComponentMap: Record<string, SettingComponent> = {
  slider: "slider",
  switch: "switch",
  radio: "radio",
  dropdown: "select",
  input: "input",
  number: "number",
  color: "color",
  checkboxgroup: "multiselect",
  separator: "separator",
};

export function metaToSettingDef(key: string, info?: OptionInfoMeta): SettingDef {
  if (!info) return { key, label: key, component: "input" };
  return {
    key,
    label: info.label || key,
    component: metaComponentMap[info.component] ?? "input",
    defaultValue: info.default,
    min: info.component_args.minimum,
    max: info.component_args.maximum,
    step: info.component_args.step,
    precision: info.component_args.precision,
    choices: info.component_args.choices,
    description: getParamHelpPlain(info.label) || undefined,
    isSecret: info.is_secret || undefined,
  };
}
