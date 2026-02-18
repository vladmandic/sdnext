export interface ControlRequestUnit {
  process: string;
  model: string;
  strength: number;
  start: number;
  end: number;
  override?: string;
  unit_type?: string;
  mode?: string;
  guess?: boolean;
  factor?: number;
  attention?: string;
  fidelity?: number;
  query_weight?: number;
  adain_weight?: number;
  image?: string;
}

export interface ControlRequest {
  prompt: string;
  negative_prompt?: string;
  sampler_name?: string;
  steps?: number;
  width_before?: number;
  height_before?: number;
  cfg_scale?: number;
  seed?: number;
  batch_size?: number;
  batch_count?: number;
  denoising_strength?: number;
  enable_hr?: boolean;
  hr_upscaler?: string;
  hr_scale?: number;
  hr_second_pass_steps?: number;
  hr_resize_x?: number;
  hr_resize_y?: number;
  hr_denoising_strength?: number;
  hr_resize_mode?: number;
  hr_resize_context?: string;
  hr_force?: boolean;
  cfg_end?: number;
  diffusers_guidance_rescale?: number;
  image_cfg_scale?: number;
  pag_scale?: number;
  pag_adaptive?: number;
  subseed?: number;
  subseed_strength?: number;
  refiner_steps?: number;
  refiner_start?: number;
  refiner_prompt?: string;
  refiner_negative?: string;
  clip_skip?: number;
  vae_type?: string;
  tiling?: boolean;
  hidiffusion?: boolean;
  hdr_mode?: number;
  hdr_brightness?: number;
  hdr_sharpen?: number;
  hdr_color?: number;
  hdr_clamp?: boolean;
  hdr_boundary?: number;
  hdr_threshold?: number;
  hdr_maximize?: boolean;
  hdr_max_center?: number;
  hdr_max_boundary?: number;
  hdr_tint_ratio?: number;
  detailer_enabled?: boolean;
  detailer_prompt?: string;
  detailer_negative?: string;
  detailer_steps?: number;
  detailer_strength?: number;
  detailer_resolution?: number;
  detailer_segmentation?: boolean;
  detailer_include_detections?: boolean;
  detailer_merge?: boolean;
  detailer_sort?: boolean;
  detailer_classes?: string;
  input_type?: number;
  inputs?: string[];
  inits?: string[];
  mask?: string;
  mask_blur?: number;
  inpaint_full_res?: boolean;
  inpaint_full_res_padding?: number;
  inpainting_mask_invert?: number;
  resize_mode_before?: number;
  ip_adapter?: Array<{
    adapter: string;
    images: string[];
    masks?: string[];
    scale: number;
    start: number;
    end: number;
    crop: boolean;
  }>;
  init_control?: string[];
  control?: ControlRequestUnit[];
  extra?: Record<string, unknown>;
  script_name?: string;
  script_args?: unknown[];
  send_images?: boolean;
  save_images?: boolean;
  alwayson_scripts?: Record<string, unknown>;
}

export interface ControlResponse {
  images: string[];
  processed: string[];
  params: Record<string, unknown>;
  info: string;
}

export interface GenerationInfo {
  prompt: string;
  negative_prompt: string;
  seed: number;
  subseed: number;
  width: number;
  height: number;
  sampler_name: string;
  cfg_scale: number;
  steps: number;
  batch_size: number;
  model: string;
  model_hash: string;
  job_timestamp: string;
  [key: string]: unknown;
}
