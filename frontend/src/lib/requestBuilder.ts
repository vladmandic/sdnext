import { useGenerationStore } from "@/stores/generationStore";
import { useAdapterStore } from "@/stores/adapterStore";
import { useScriptStore } from "@/stores/scriptStore";
import { useControlStore } from "@/stores/controlStore";
import { fileToBase64 } from "@/lib/image";
import type { Txt2ImgRequest } from "@/api/types/generation";

export async function buildTxt2ImgRequest(): Promise<Txt2ImgRequest> {
  const gen = useGenerationStore.getState();
  const adapters = useAdapterStore.getState();
  const scripts = useScriptStore.getState();
  const control = useControlStore.getState();

  const request: Txt2ImgRequest = {
    prompt: gen.prompt,
    negative_prompt: gen.negativePrompt,
    sampler_name: gen.sampler,
    steps: gen.steps,
    width: gen.width,
    height: gen.height,
    cfg_scale: gen.cfgScale,
    cfg_end: gen.cfgEnd,
    diffusers_guidance_rescale: gen.guidanceRescale,
    image_cfg_scale: gen.imageCfgScale,
    diffusers_pag_scale: gen.pagScale,
    diffusers_pag_adaptive: gen.pagAdaptive,
    seed: gen.seed,
    subseed: gen.subseed,
    subseed_strength: gen.subseedStrength,
    batch_size: gen.batchSize,
    n_iter: gen.batchCount,
    denoising_strength: gen.denoisingStrength,
    enable_hr: gen.hiresEnabled,
    hr_upscaler: gen.hiresUpscaler,
    hr_scale: gen.hiresScale,
    hr_second_pass_steps: gen.hiresSteps,
    hr_denoising_strength: gen.hiresDenoising,
    hr_sampler_name: gen.hiresSampler || undefined,
    hr_force: gen.hiresForce,
    hr_resize_mode: gen.hiresResizeMode,
    hr_resize_x: gen.hiresResizeX,
    hr_resize_y: gen.hiresResizeY,
    refiner_steps: gen.refinerSteps,
    refiner_start: gen.refinerStart,
    refiner_prompt: gen.refinerPrompt || undefined,
    refiner_negative: gen.refinerNegative || undefined,
    clip_skip: gen.clipSkip,
    vae_type: gen.vaeType,
    tiling: gen.tiling,
    hidiffusion: gen.hidiffusion,
    hdr_mode: gen.hdrMode,
    hdr_brightness: gen.hdrBrightness,
    hdr_sharpen: gen.hdrSharpen,
    hdr_color: gen.hdrColor,
    hdr_clamp: gen.hdrClamp,
    hdr_boundary: gen.hdrBoundary,
    hdr_threshold: gen.hdrThreshold,
    hdr_maximize: gen.hdrMaximize,
    hdr_max_center: gen.hdrMaxCenter,
    hdr_max_boundary: gen.hdrMaxBoundary,
    hdr_tint_ratio: gen.hdrTintRatio,
    override_settings: {
      schedulers_sigma: gen.sigmaMethod,
      schedulers_timestep_spacing: gen.timestepSpacing,
      schedulers_beta_schedule: gen.betaSchedule,
      schedulers_prediction_type: gen.predictionMethod,
      schedulers_shift: gen.flowShift,
      schedulers_base_shift: gen.baseShift,
      schedulers_max_shift: gen.maxShift,
      schedulers_sigma_adjust: gen.sigmaAdjust,
      schedulers_sigma_adjust_min: gen.sigmaAdjustStart,
      schedulers_sigma_adjust_max: gen.sigmaAdjustEnd,
      schedulers_use_thresholding: gen.thresholding,
      schedulers_dynamic_shift: gen.dynamic,
      schedulers_rescale_betas: gen.rescale,
      schedulers_use_loworder: gen.lowOrder,
      ...(gen.timestepsOverride ? { schedulers_timesteps: gen.timestepsOverride } : {}),
    },
  };

  // Detailer
  if (gen.detailerEnabled) {
    request.detailer_enabled = true;
    request.detailer_prompt = gen.detailerPrompt;
    request.detailer_negative = gen.detailerNegative;
    request.detailer_steps = gen.detailerSteps;
    request.detailer_strength = gen.detailerStrength;
    request.detailer_resolution = gen.detailerResolution;
    request.detailer_segmentation = gen.detailerSegmentation;
    request.detailer_include_detections = gen.detailerIncludeDetections;
    request.detailer_merge = gen.detailerMerge;
    request.detailer_sort = gen.detailerSort;
    request.detailer_classes = gen.detailerClasses || undefined;
    request.override_settings = {
      ...request.override_settings,
      detailer_models: gen.detailerModels,
      detailer_max_detected: gen.detailerMaxDetected,
      detailer_padding: gen.detailerPadding,
      detailer_blur: gen.detailerBlur,
      detailer_confidence: gen.detailerConfidence,
      detailer_iou: gen.detailerIou,
      detailer_min_size: gen.detailerMinSize,
      detailer_max_size: gen.detailerMaxSize,
      detailer_renoise: gen.detailerRenoise,
      detailer_renoise_end: gen.detailerRenoiseEnd,
    };
  }

  // IP-Adapter
  const activeAdapters = adapters.units
    .slice(0, adapters.activeUnits)
    .filter((u) => u.adapter !== "None" && u.images.length > 0);
  if (activeAdapters.length > 0) {
    request.ip_adapter = await Promise.all(
      activeAdapters.map(async (u) => ({
        adapter: u.adapter,
        scale: u.scale,
        crop: u.crop,
        start: u.start,
        end: u.end,
        images: await Promise.all(u.images.map(fileToBase64)),
        masks: u.masks.length > 0 ? await Promise.all(u.masks.map(fileToBase64)) : undefined,
      })),
    );
  }

  // Scripts
  if (scripts.selectedScript) {
    request.script_name = scripts.selectedScript;
    request.script_args = scripts.scriptArgs;
  }
  const alwaysOnKeys = Object.keys(scripts.alwaysOnOverrides);
  if (alwaysOnKeys.length > 0) {
    request.alwayson_scripts = {};
    for (const name of alwaysOnKeys) {
      request.alwayson_scripts[name] = { args: scripts.alwaysOnOverrides[name] };
    }
  }

  // Control: ControlNet units as alwayson_scripts
  const enabledUnits = control.units.filter((u) => u.enabled && u.image);
  if (enabledUnits.length > 0) {
    const controlArgs = await Promise.all(
      enabledUnits.map(async (u) => ({
        enabled: true,
        processor: u.processor,
        model: u.model,
        strength: u.strength,
        start: u.start,
        end: u.end,
        image: u.image ? await fileToBase64(u.image) : null,
      })),
    );
    request.alwayson_scripts = {
      ...request.alwayson_scripts,
      controlnet: { args: controlArgs },
    };
  }

  // User override settings (merged last to take priority)
  if (Object.keys(gen.overrideSettings).length > 0) {
    request.override_settings = {
      ...request.override_settings,
      ...gen.overrideSettings,
    };
  }

  return request;
}
