import type { OptionInfoMeta } from "@/api/types/settings";

export type SettingComponent = "slider" | "switch" | "select" | "radio" | "input" | "number" | "color" | "separator" | "multiselect";

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
      { key: "sd_model_checkpoint", label: "Checkpoint", component: "select", description: "Active model checkpoint (populated dynamically)" },
      { key: "sd_model_refiner", label: "Refiner", component: "select", description: "Refiner model for two-stage generation", defaultValue: "None" },
      { key: "sd_checkpoint_autoload", label: "Auto-load model", component: "switch", defaultValue: true, description: "Automatically load selected model on startup" },
      { key: "diffusers_pipeline", label: "Pipeline", component: "select", description: "Diffusers pipeline type" },
    ],
  },
  {
    id: "offload",
    title: "Model Offloading",
    settings: [
      { key: "diffusers_offload_mode", label: "Offload mode", component: "select", defaultValue: "none", description: "Memory offloading strategy" },
    ],
  },
  {
    id: "vae_encoder",
    title: "VAE",
    settings: [
      { key: "sd_vae", label: "VAE model", component: "select", description: "VAE model override (populated dynamically)", defaultValue: "Automatic" },
      { key: "no_half_vae", label: "Full precision VAE", component: "switch", defaultValue: false, description: "Run VAE in full precision (fixes NaN issues)" },
      { key: "diffusers_vae_upcast", label: "VAE upcast", component: "select", defaultValue: "default", description: "Upcast VAE to float32 for computation" },
      { key: "diffusers_vae_slicing", label: "VAE slicing", component: "switch", defaultValue: false, description: "Process VAE in slices to save memory" },
      { key: "diffusers_vae_tiling", label: "VAE tiling", component: "switch", defaultValue: false, description: "Use tiled VAE for large images" },
    ],
  },
  {
    id: "text_encoder",
    title: "Text Encoder",
    settings: [
      { key: "sd_text_encoder", label: "Text encoder model", component: "select", description: "Text encoder override (populated dynamically)" },
      { key: "prompt_attention", label: "Prompt attention", component: "select", description: "Prompt attention parser implementation" },
      { key: "prompt_mean_norm", label: "Mean normalization", component: "switch", defaultValue: true, description: "Normalize prompt embeddings with mean" },
      { key: "comma_padding_backtrack", label: "Comma padding", component: "number", min: 0, max: 74, defaultValue: 20, description: "Backtrack padding when splitting long prompts" },
    ],
  },
  {
    id: "cuda",
    title: "Compute",
    settings: [
      { key: "precision", label: "Precision", component: "select", defaultValue: "Autocast", description: "Computation precision mode" },
      { key: "cuda_dtype", label: "CUDA dtype", component: "select", description: "Default tensor dtype for CUDA" },
      { key: "no_half", label: "Disable half precision", component: "switch", defaultValue: false, description: "Run entire model in full precision" },
      { key: "cross_attention_optimization", label: "Cross attention", component: "select", defaultValue: "Automatic", description: "Cross attention optimization method" },
      { key: "cuda_compile_backend", label: "Compile backend", component: "select", defaultValue: "none", description: "Torch compile backend for inference optimization" },
    ],
  },
  {
    id: "saving-images",
    title: "Image Options",
    settings: [
      { key: "samples_save", label: "Save images", component: "switch", defaultValue: true, description: "Save generated images to disk" },
      { key: "samples_format", label: "Image format", component: "select", defaultValue: "jpg", description: "Output image format" },
      { key: "samples_filename_pattern", label: "Filename pattern", component: "input", defaultValue: "", description: "Custom filename pattern (empty = default)" },
      { key: "jpeg_quality", label: "JPEG quality", component: "slider", min: 1, max: 100, step: 1, defaultValue: 85, description: "Quality for JPEG/WebP output" },
      { key: "save_images_add_number", label: "Add number", component: "switch", defaultValue: true, description: "Add sequence number to filenames" },
      { key: "grid_save", label: "Save grids", component: "switch", defaultValue: true, description: "Save image grids when batch > 1" },
      { key: "grid_format", label: "Grid format", component: "select", defaultValue: "jpg", description: "Grid image format" },
      { key: "use_save_to_dirs_for_ui", label: "Save to subdirs", component: "switch", defaultValue: false, description: "Save images into date-based subdirectories" },
    ],
  },
  {
    id: "live-preview",
    title: "Live Previews",
    settings: [
      { key: "show_progress_every_n_steps", label: "Preview interval", component: "number", min: 0, max: 50, step: 1, defaultValue: 1, description: "Show preview every N steps (0 = disabled)" },
      { key: "show_progress_type", label: "Preview method", component: "select", defaultValue: "TAESD", description: "Live preview decode method" },
      { key: "live_previews_enable", label: "Enable previews", component: "switch", defaultValue: true, description: "Show live previews during generation" },
    ],
  },
  {
    id: "postprocessing",
    title: "Postprocessing",
    settings: [
      { key: "postprocessing_enable_in_main_ui", label: "Enable in main UI", component: "switch", defaultValue: false, description: "Show postprocessing options in main generation UI" },
      { key: "upscaler_for_img2img", label: "Img2img upscaler", component: "select", description: "Upscaler for img2img resize (populated dynamically)" },
      { key: "face_restoration", label: "Face restoration", component: "switch", defaultValue: false, description: "Apply face restoration to generated images" },
      { key: "face_restoration_model", label: "Face model", component: "select", defaultValue: "CodeFormer", description: "Face restoration model" },
      { key: "code_former_weight", label: "CodeFormer weight", component: "slider", min: 0, max: 1, step: 0.05, defaultValue: 0.2, description: "CodeFormer fidelity vs enhancement weight" },
    ],
  },
];

export const QUICK_SETTINGS_GROUPS = [
  { title: "Model", keys: ["sd_model_checkpoint", "sd_vae", "sd_model_refiner", "sd_model_dict"] },
  { title: "Performance", keys: ["cross_attention_optimization", "cuda_compile_backend", "cuda_compile_mode", "diffusers_offload_mode"] },
  { title: "Output", keys: ["samples_save", "save_init_img", "image_watermark_enabled", "mask_apply_overlay", "batch_frame_mode"] },
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
    isSecret: info.is_secret || undefined,
  };
}
