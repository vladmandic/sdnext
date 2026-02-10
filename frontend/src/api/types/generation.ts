export interface Txt2ImgRequest {
  prompt: string;
  negative_prompt?: string;
  sampler_name?: string;
  steps?: number;
  width?: number;
  height?: number;
  cfg_scale?: number;
  seed?: number;
  batch_size?: number;
  n_iter?: number;
  denoising_strength?: number;
  enable_hr?: boolean;
  hr_upscaler?: string;
  hr_scale?: number;
  hr_second_pass_steps?: number;
  hr_resize_x?: number;
  hr_resize_y?: number;
  hr_denoising_strength?: number;
  hr_sampler_name?: string;
  cfg_end?: number;
  diffusers_guidance_rescale?: number;
  image_cfg_scale?: number;
  diffusers_pag_scale?: number;
  diffusers_pag_adaptive?: number;
  subseed?: number;
  subseed_strength?: number;
  hr_force?: boolean;
  hr_resize_mode?: number;
  hr_resize_context?: string;
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
  control_units?: Array<{
    enabled: boolean;
    unit_type: string;
    processor: string;
    model: string;
    image: string;
    strength: number;
    start: number;
    end: number;
    guess?: boolean;
    factor?: number;
    attention?: string;
    fidelity?: number;
    query_weight?: number;
    adain_weight?: number;
  }>;
  script_name?: string;
  script_args?: unknown[];
  send_images?: boolean;
  save_images?: boolean;
  alwayson_scripts?: Record<string, unknown>;
  override_settings?: Record<string, unknown>;
}

export interface Img2ImgRequest extends Txt2ImgRequest {
  init_images?: string[];
  mask?: string;
  mask_blur?: number;
  denoising_strength?: number;
  resize_mode?: number;
  image_cfg_scale?: number;
  inpaint_full_res?: boolean;
  inpaint_full_res_padding?: number;
  inpainting_fill?: number;
  inpainting_mask_invert?: number;
}

export interface GenerationResponse {
  images: string[];
  parameters: Record<string, unknown>;
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
