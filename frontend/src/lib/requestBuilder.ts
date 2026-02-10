import { useGenerationStore } from "@/stores/generationStore";
import type { GenerationResult } from "@/stores/generationStore";
import { useScriptStore } from "@/stores/scriptStore";
import { useControlStore } from "@/stores/controlStore";
import { useImg2ImgStore } from "@/stores/img2imgStore";
import { useCanvasStore, type ImageLayer } from "@/stores/canvasStore";
import { useUiStore } from "@/stores/uiStore";
import { fileToBase64 } from "@/lib/image";
import { exportMaskToBase64 } from "@/lib/exportMask";
import { flattenCanvas } from "@/lib/flattenCanvas";
import type { Txt2ImgRequest, Img2ImgRequest, GenerationInfo } from "@/api/types/generation";

export async function buildTxt2ImgRequest(): Promise<Txt2ImgRequest> {
  const gen = useGenerationStore.getState();
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
      detailer_max: gen.detailerMaxDetected,
      detailer_padding: gen.detailerPadding,
      detailer_blur: gen.detailerBlur,
      detailer_conf: gen.detailerConfidence,
      detailer_iou: gen.detailerIou,
      detailer_min_size: gen.detailerMinSize,
      detailer_max_size: gen.detailerMaxSize,
      detailer_sigma_adjust: gen.detailerRenoise,
      detailer_sigma_adjust_max: gen.detailerRenoiseEnd,
    };
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

  // Control units: partition by type — IP-adapter vs control types
  const enabledIPUnits = control.units.filter((u) => u.enabled && u.unitType === "ip" && u.images.length > 0);
  const TYPE_MAP: Record<string, string> = { controlnet: "controlnet", t2i: "t2i adapter", xs: "xs", lite: "lite", reference: "reference" };
  const enabledControlUnits = control.units.filter((u) => u.enabled && u.unitType !== "ip" && u.unitType !== "asset" && u.image);
  const enabledAssetUnits = control.units.filter((u) => u.enabled && u.unitType === "asset" && u.image);

  if (enabledIPUnits.length > 0) {
    request.ip_adapter = await Promise.all(
      enabledIPUnits.map(async (u) => ({
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

  if (enabledControlUnits.length > 0) {
    request.control_units = await Promise.all(
      enabledControlUnits.map(async (u) => ({
        enabled: true,
        unit_type: TYPE_MAP[u.unitType] ?? u.unitType,
        processor: u.processor,
        model: u.model,
        strength: u.strength,
        start: u.start,
        end: u.end,
        image: u.image ? await fileToBase64(u.image) : "",
        mode: u.mode,
        ...(u.unitType === "controlnet" ? { guess: u.guess } : {}),
        ...(u.unitType === "t2i" ? { factor: u.factor } : {}),
        ...(u.unitType === "reference" ? { attention: u.attention, fidelity: u.fidelity, query_weight: u.queryWeight, adain_weight: u.adainWeight } : {}),
      })),
    );
  }

  if (enabledAssetUnits.length > 0) {
    request.init_control = await Promise.all(enabledAssetUnits.map((u) => fileToBase64(u.image!)));
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

export async function buildImg2ImgRequest(): Promise<Img2ImgRequest> {
  const baseRequest = await buildTxt2ImgRequest();
  const img2img = useImg2ImgStore.getState();
  const gen = useGenerationStore.getState();
  const canvas = useCanvasStore.getState();

  const width = gen.width;
  const height = gen.height;

  // Flatten all image layers into a single composite
  const imageLayers = canvas.layers.filter((l) => l.type === "image") as ImageLayer[];
  const flattenedBase64 = await flattenCanvas(imageLayers, width, height);

  // Export mask from painted strokes if no explicit maskData
  let maskData = img2img.maskData;
  if (!maskData && img2img.maskLines.length > 0) {
    maskData = exportMaskToBase64(img2img.maskLines, width, height);
  }

  return {
    ...baseRequest,
    width,
    height,
    init_images: flattenedBase64 ? [flattenedBase64] : [],
    resize_mode: img2img.resizeMode,
    mask: maskData || undefined,
    mask_blur: img2img.maskBlur,
    inpaint_full_res: img2img.inpaintFullRes,
    inpaint_full_res_padding: img2img.inpaintFullResPadding,
    inpainting_mask_invert: img2img.inpaintingMaskInvert ? 1 : 0,
  };
}

/** Restore generation store state from a previous result. */
export function restoreFromResult(result: GenerationResult): void {
  const p = result.parameters;

  // Parse the info JSON for resolved values (actual seed used, etc.)
  let info: GenerationInfo | null = null;
  try { info = JSON.parse(result.info) as GenerationInfo; }
  catch { /* ignore */ }

  const overrides = (p.override_settings ?? {}) as Record<string, unknown>;

  const num = (v: unknown, fallback: number) => typeof v === "number" ? v : fallback;
  const str = (v: unknown, fallback: string) => typeof v === "string" ? v : fallback;
  const bool = (v: unknown, fallback: boolean) => typeof v === "boolean" ? v : fallback;

  useGenerationStore.getState().setParams({
    // Prompt
    prompt: str(p.prompt, ""),
    negativePrompt: str(p.negative_prompt, ""),

    // Sampler
    sampler: str(p.sampler_name, "Euler"),
    steps: num(p.steps, 20),

    // Resolution
    width: num(p.width, 1024),
    height: num(p.height, 1024),

    // Batch
    batchSize: num(p.batch_size, 1),
    batchCount: num(p.n_iter, 1),

    // Guidance
    cfgScale: num(p.cfg_scale, 7),
    cfgEnd: num(p.cfg_end, 1),
    guidanceRescale: num(p.diffusers_guidance_rescale, 0),
    imageCfgScale: num(p.image_cfg_scale, 6),
    pagScale: num(p.diffusers_pag_scale, 0),
    pagAdaptive: num(p.diffusers_pag_adaptive, 0.5),
    denoisingStrength: num(p.denoising_strength, 0.5),

    // Seed — use resolved values from info when available
    seed: num(info?.seed ?? p.seed, -1),
    subseed: num(info?.subseed ?? p.subseed, -1),
    subseedStrength: num(p.subseed_strength, 0),

    // Hires
    hiresEnabled: bool(p.enable_hr, false),
    hiresUpscaler: str(p.hr_upscaler, "Latent"),
    hiresScale: num(p.hr_scale, 2),
    hiresSteps: num(p.hr_second_pass_steps, 0),
    hiresDenoising: num(p.hr_denoising_strength, 0.5),
    hiresSampler: str(p.hr_sampler_name, ""),
    hiresForce: bool(p.hr_force, false),
    hiresResizeMode: num(p.hr_resize_mode, 0),
    hiresResizeX: num(p.hr_resize_x, 0),
    hiresResizeY: num(p.hr_resize_y, 0),

    // Refiner
    refinerSteps: num(p.refiner_steps, 0),
    refinerStart: num(p.refiner_start, 0),
    refinerPrompt: str(p.refiner_prompt, ""),
    refinerNegative: str(p.refiner_negative, ""),

    // Advanced
    clipSkip: num(p.clip_skip, 1),
    vaeType: str(p.vae_type, "Full"),
    tiling: bool(p.tiling, false),
    hidiffusion: bool(p.hidiffusion, false),

    // HDR corrections
    hdrMode: num(p.hdr_mode, 0),
    hdrBrightness: num(p.hdr_brightness, 0),
    hdrSharpen: num(p.hdr_sharpen, 0),
    hdrColor: num(p.hdr_color, 0),
    hdrClamp: bool(p.hdr_clamp, false),
    hdrBoundary: num(p.hdr_boundary, 4.0),
    hdrThreshold: num(p.hdr_threshold, 0.95),
    hdrMaximize: bool(p.hdr_maximize, false),
    hdrMaxCenter: num(p.hdr_max_center, 0.6),
    hdrMaxBoundary: num(p.hdr_max_boundary, 1.0),
    hdrTintRatio: num(p.hdr_tint_ratio, 0),

    // Detailer
    detailerEnabled: bool(p.detailer_enabled, false),
    detailerPrompt: str(p.detailer_prompt, ""),
    detailerNegative: str(p.detailer_negative, ""),
    detailerSteps: num(p.detailer_steps, 10),
    detailerStrength: num(p.detailer_strength, 0.3),
    detailerResolution: num(p.detailer_resolution, 1024),
    detailerSegmentation: bool(p.detailer_segmentation, false),
    detailerIncludeDetections: bool(p.detailer_include_detections, false),
    detailerMerge: bool(p.detailer_merge, false),
    detailerSort: bool(p.detailer_sort, false),
    detailerClasses: str(p.detailer_classes, ""),

    // Scheduler overrides
    sigmaMethod: str(overrides.schedulers_sigma, "default"),
    timestepSpacing: str(overrides.schedulers_timestep_spacing, "default"),
    betaSchedule: str(overrides.schedulers_beta_schedule, "default"),
    predictionMethod: str(overrides.schedulers_prediction_type, "default"),
    flowShift: num(overrides.schedulers_shift, 3),
    baseShift: num(overrides.schedulers_base_shift, 0.5),
    maxShift: num(overrides.schedulers_max_shift, 1.15),
    sigmaAdjust: num(overrides.schedulers_sigma_adjust, 1.0),
    sigmaAdjustStart: num(overrides.schedulers_sigma_adjust_min, 0.2),
    sigmaAdjustEnd: num(overrides.schedulers_sigma_adjust_max, 1.0),
    thresholding: bool(overrides.schedulers_use_thresholding, false),
    dynamic: bool(overrides.schedulers_dynamic_shift, false),
    rescale: bool(overrides.schedulers_rescale_betas, false),
    lowOrder: bool(overrides.schedulers_use_loworder, true),
    timestepsOverride: str(overrides.schedulers_timesteps, ""),
    timestepsPreset: "None",

    // Detailer overrides
    ...(p.detailer_enabled ? {
      detailerModels: Array.isArray(overrides.detailer_models) ? overrides.detailer_models as string[] : ["face-yolo8n"],
      detailerMaxDetected: num(overrides.detailer_max, 2),
      detailerPadding: num(overrides.detailer_padding, 20),
      detailerBlur: num(overrides.detailer_blur, 10),
      detailerConfidence: num(overrides.detailer_conf, 0.6),
      detailerIou: num(overrides.detailer_iou, 0.5),
      detailerMinSize: num(overrides.detailer_min_size, 0.0),
      detailerMaxSize: num(overrides.detailer_max_size, 1.0),
      detailerRenoise: num(overrides.detailer_sigma_adjust, 1.0),
      detailerRenoiseEnd: num(overrides.detailer_sigma_adjust_max, 1.0),
    } : {}),
  });

  // Restore input image and mask if present (img2img history)
  if (result.inputImage) {
    const w = num(p.width, 1024);
    const h = num(p.height, 1024);
    useCanvasStore.getState().restoreImageLayer(result.inputImage, w, h);
    useUiStore.getState().setGenerationMode("img2img");

    if (result.inputMask && result.inputMask.length > 0) {
      const img2imgState = useImg2ImgStore.getState();
      img2imgState.clearMask();
      for (const line of result.inputMask) {
        img2imgState.addMaskLine(line);
      }
    }
  }

  // Restore control units if present
  if (result.controlUnits && result.controlUnits.length > 0) {
    useControlStore.getState().restoreUnits(result.controlUnits);
  }
}
