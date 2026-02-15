// Phase 1

export interface ModelModule {
  name: string;
  cls: string;
  device: string | null;
  dtype: string | null;
  quant: string | null;
  params: number;
  modules: number;
  config: Record<string, unknown> | null;
}

export interface ModelAnalysis {
  name: string;
  type: string;
  class: string;
  hash: string | null;
  size: number;
  mtime: string | null;
  meta: Record<string, unknown>;
  modules: ModelModule[];
}

export interface ModelSaveRequest {
  name: string;
  path?: string;
  shard?: string;
  overwrite?: boolean;
}

export interface ModelListDetail {
  model_name: string;
  filename: string;
  type: string;
  detected_type: string;
  pipeline: string | null;
  hash: string | null;
  size: number;
  mtime: string | null;
}

// Phase 2

export interface HfModelResult {
  id: string;
  pipeline_tag: string;
  tags: string;
  downloads: number;
  last_modified: string;
  url: string;
}

export interface HfDownloadRequest {
  hub_id: string;
  token?: string;
  variant?: string;
  revision?: string;
  mirror?: string;
  custom_pipeline?: string;
}

export interface CivitaiDownloadRequest {
  url: string;
  name?: string;
  path?: string;
  model_type?: string;
  token?: string;
}

export interface CivitMetadataScanResult {
  name: string;
  id: number | null;
  type: string;
  code: number;
  hash: string;
  size: number;
  note: string;
}

export interface CivitMetadataUpdateResult {
  file: string | null;
  id: number | null;
  name: string | null;
  sha: string | null;
  versions: number | null;
  latest: string | null;
  status: string | null;
}

// Phase 3

export interface MergeMethodsInfo {
  methods: string[];
  beta_methods: string[];
  triple_methods: string[];
  docs: Record<string, string>;
  presets: Record<string, number[]>;
  sdxl_presets: Record<string, number[]>;
}

export interface MergeRequest {
  custom_name: string;
  primary_model_name: string;
  secondary_model_name: string;
  merge_mode: string;
  tertiary_model_name?: string;
  alpha?: number;
  beta?: number;
  alpha_preset?: string;
  alpha_preset_lambda?: number;
  alpha_base?: string;
  alpha_in_blocks?: string;
  alpha_mid_block?: string;
  alpha_out_blocks?: string;
  beta_preset?: string;
  beta_preset_lambda?: number;
  beta_base?: string;
  beta_in_blocks?: string;
  beta_mid_block?: string;
  beta_out_blocks?: string;
  precision?: string;
  checkpoint_format?: string;
  save_metadata?: boolean;
  weights_clip?: boolean;
  prune?: boolean;
  re_basin?: boolean;
  re_basin_iterations?: number;
  device?: string;
  unload?: boolean;
  overwrite?: boolean;
  bake_in_vae?: string;
}

export interface ReplaceRequest {
  model_type: string;
  model_name: string;
  custom_name: string;
  comp_unet?: string;
  comp_vae?: string;
  comp_te1?: string;
  comp_te2?: string;
  precision?: string;
  comp_scheduler?: string;
  comp_prediction?: string;
  comp_lora?: string;
  comp_fuse?: number;
  meta_author?: string;
  meta_version?: string;
  meta_license?: string;
  meta_desc?: string;
  meta_hint?: string;
  create_diffusers?: boolean;
  create_safetensors?: boolean;
  debug?: boolean;
}

// Phase 4

export interface LoaderComponent {
  id: number;
  name: string;
  loadable: boolean;
  default: string | null;
  class_name: string;
  local: string | null;
  remote: string | null;
  dtype: string | null;
  quant: boolean | null;
}

export interface LoaderComponentsResponse {
  class: string;
  repo: string | null;
  components: LoaderComponent[];
}

export interface LoaderLoadRequest {
  model_type: string;
  repo: string;
  components?: Array<{
    id: number;
    local?: string;
    remote?: string;
    dtype?: string;
    quant?: boolean;
  }>;
}

export interface LoraExtractRequest {
  filename: string;
  max_rank?: number;
  auto_rank?: boolean;
  rank_ratio?: number;
  modules?: string[];
  overwrite?: boolean;
}
