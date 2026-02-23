import { useGenerationStore } from "@/stores/generationStore";
import type { GenerationResult } from "@/stores/generationStore";
import { useScriptStore } from "@/stores/scriptStore";
import { useControlStore, resolveUnitImage } from "@/stores/controlStore";
import { useImg2ImgStore } from "@/stores/img2imgStore";
import { useCanvasStore, type ImageLayer } from "@/stores/canvasStore";
import { useUiStore } from "@/stores/uiStore";
import { exportMask } from "@/lib/exportMask";
import { flattenCanvas, compositeControlImage } from "@/lib/flattenCanvas";
import { uploadFile, uploadFiles, uploadBlob } from "@/lib/upload";
import { REFERENCE_HEIGHT } from "@/canvas/useControlFrameLayout";
import { resolveGenerationSize } from "@/lib/sizeCompute";
import type { SizeMode } from "@/lib/sizeCompute";
import type { ControlRequest, GenerationInfo } from "@/api/types/generation";

export interface BuildResult {
  request: ControlRequest;
  inputBlob?: Blob;
}

export async function buildControlRequest(): Promise<BuildResult> {
  const gen = useGenerationStore.getState();
  const scripts = useScriptStore.getState();
  const control = useControlStore.getState();
  const img2img = useImg2ImgStore.getState();
  const canvas = useCanvasStore.getState();
  const ui = useUiStore.getState();

  const isImg2Img = canvas.getImageLayers().length > 0;

  const request: ControlRequest = {
    prompt: gen.prompt,
    negative_prompt: gen.negativePrompt,
    sampler_name: gen.sampler,
    steps: gen.steps,
    width_before: gen.width,
    height_before: gen.height,
    cfg_scale: gen.cfgScale,
    save_images: true,
    cfg_end: gen.cfgEnd,
    diffusers_guidance_rescale: gen.guidanceRescale,
    image_cfg_scale: gen.imageCfgScale,
    pag_scale: gen.pagScale,
    pag_adaptive: gen.pagAdaptive,
    seed: gen.seed,
    subseed: gen.subseed,
    subseed_strength: gen.subseedStrength,
    batch_size: gen.batchSize,
    batch_count: gen.batchCount,
    denoising_strength: gen.denoisingStrength,
    enable_hr: gen.hiresEnabled,
    hr_upscaler: gen.hiresUpscaler,
    hr_scale: gen.hiresScale,
    hr_second_pass_steps: gen.hiresSteps,
    hr_denoising_strength: gen.hiresDenoising,
    hr_force: gen.hiresForce,
    hr_resize_mode: gen.hiresResizeMode,
    hr_resize_x: gen.hiresResizeX,
    hr_resize_y: gen.hiresResizeY,
    hr_resize_context: gen.hiresResizeContext,
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
    hdr_color_picker: gen.hdrColorPicker,
    hdr_tint_ratio: gen.hdrTintRatio,
    grading_brightness: gen.gradingBrightness,
    grading_contrast: gen.gradingContrast,
    grading_saturation: gen.gradingSaturation,
    grading_hue: gen.gradingHue,
    grading_gamma: gen.gradingGamma,
    grading_sharpness: gen.gradingSharpness,
    grading_color_temp: gen.gradingColorTemp,
    grading_shadows: gen.gradingShadows,
    grading_midtones: gen.gradingMidtones,
    grading_highlights: gen.gradingHighlights,
    grading_clahe_clip: gen.gradingClaheClip,
    grading_clahe_grid: gen.gradingClaheGrid,
    grading_shadows_tint: gen.gradingShadowsTint,
    grading_highlights_tint: gen.gradingHighlightsTint,
    grading_split_tone_balance: gen.gradingSplitToneBalance,
    grading_vignette: gen.gradingVignette,
    grading_grain: gen.gradingGrain,
    grading_lut_file: gen.gradingLutFile || undefined,
    grading_lut_strength: gen.gradingLutStrength,
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
    ...(gen.freeuEnabled ? {
      freeu_enabled: true,
      freeu_b1: gen.freeuB1,
      freeu_b2: gen.freeuB2,
      freeu_s1: gen.freeuS1,
      freeu_s2: gen.freeuS2,
    } : {}),
    ...(gen.hypertileUnetEnabled ? {
      hypertile_unet_enabled: true,
      hypertile_hires_only: gen.hypertileHiresOnly,
      hypertile_unet_tile: gen.hypertileUnetTile,
      hypertile_unet_min_tile: gen.hypertileUnetMinTile,
      hypertile_unet_swap_size: gen.hypertileUnetSwapSize,
      hypertile_unet_depth: gen.hypertileUnetDepth,
    } : {}),
    ...(gen.hypertileVaeEnabled ? {
      hypertile_vae_enabled: true,
      hypertile_vae_tile: gen.hypertileVaeTile,
      hypertile_vae_swap_size: gen.hypertileVaeSwapSize,
    } : {}),
    ...(gen.teacacheEnabled ? {
      teacache_enabled: true,
      teacache_thresh: gen.teacacheThresh,
    } : {}),
    ...(gen.tokenMergingMethod !== "None" ? {
      token_merging_method: gen.tokenMergingMethod,
      tome_ratio: gen.tomeRatio,
      todo_ratio: gen.todoRatio,
    } : {}),
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
    request.detailer_models = gen.detailerModels;
    request.detailer_max = gen.detailerMaxDetected;
    request.detailer_padding = gen.detailerPadding;
    request.detailer_blur = gen.detailerBlur;
    request.detailer_conf = gen.detailerConfidence;
    request.detailer_iou = gen.detailerIou;
    request.detailer_min_size = gen.detailerMinSize;
    request.detailer_max_size = gen.detailerMaxSize;
    request.detailer_sigma_adjust = gen.detailerRenoise;
    request.detailer_sigma_adjust_max = gen.detailerRenoiseEnd;
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

  // Resolve images for control units (may reference another unit's image via "unit:N")
  const controlUnitEntries = control.units
    .map((u, i) => ({ unit: u, image: resolveUnitImage(control.units, i) }))
    .filter((e) => e.unit.enabled && e.unit.unitType !== "ip" && e.unit.unitType !== "asset" && e.image);
  const assetUnitEntries = control.units
    .map((u, i) => ({ unit: u, image: resolveUnitImage(control.units, i) }))
    .filter((e) => e.unit.enabled && e.unit.unitType === "asset" && e.image);

  if (enabledIPUnits.length > 0) {
    request.ip_adapter = await Promise.all(
      enabledIPUnits.map(async (u) => ({
        adapter: u.adapter,
        scale: u.scale,
        crop: u.crop,
        start: u.start,
        end: u.end,
        images: await uploadFiles(u.images),
        masks: u.masks.length > 0 ? await uploadFiles(u.masks) : undefined,
      })),
    );
  }

  // Compute display scale for free-mode compositing
  const displayScale = gen.height > 0 ? REFERENCE_HEIGHT / gen.height : 1;

  if (controlUnitEntries.length > 0) {
    const reprocess = ui.reprocessOnGenerate;
    request.control = await Promise.all(
      controlUnitEntries.map(async (e) => {
        // When reprocess is off and a manual preview exists, send the processed image
        // as override with process=None so the backend uses it as-is.
        const hasManualPreview = !reprocess && e.unit.processedImage;
        let overrideRef: string | undefined;
        if (hasManualPreview) {
          const resp = await fetch(e.unit.processedImage!);
          const blob = await resp.blob();
          overrideRef = await uploadBlob(blob, "processed.png");
        } else if (e.unit.fitMode === "free" && e.image) {
          // Free mode: composite the image at generation resolution before uploading
          const ft = e.unit.freeTransform ?? { x: 0, y: 0, scaleX: 1, scaleY: 1, rotation: 0 };
          const composed = await compositeControlImage(e.image, ft, gen.width, gen.height, displayScale);
          overrideRef = await uploadBlob(composed, "control.png");
        } else if (e.image) {
          overrideRef = await uploadFile(e.image);
        }
        return {
          process: hasManualPreview ? "None" : e.unit.processor,
          model: e.unit.model,
          strength: e.unit.strength,
          start: e.unit.start,
          end: e.unit.end,
          override: overrideRef,
          unit_type: TYPE_MAP[e.unit.unitType] ?? e.unit.unitType,
          mode: e.unit.mode,
          ...(e.unit.unitType === "controlnet" ? { guess: e.unit.guess } : {}),
          ...(e.unit.unitType === "t2i" ? { factor: e.unit.factor } : {}),
          ...(e.unit.unitType === "reference" ? { attention: e.unit.attention, fidelity: e.unit.fidelity, query_weight: e.unit.queryWeight, adain_weight: e.unit.adainWeight } : {}),
        };
      }),
    );
  }

  if (assetUnitEntries.length > 0) {
    request.init_control = await Promise.all(assetUnitEntries.map(async (e) => {
      if (e.unit.fitMode === "free" && e.image) {
        const ft = e.unit.freeTransform ?? { x: 0, y: 0, scaleX: 1, scaleY: 1, rotation: 0 };
        const composed = await compositeControlImage(e.image, ft, gen.width, gen.height, displayScale);
        return uploadBlob(composed, "control.png");
      }
      return uploadFile(e.image!);
    }));
  }

  // img2img: add inputs, mask, inpainting params
  let inputBlob: Blob | undefined;
  if (isImg2Img) {
    const frameW = gen.width;
    const frameH = gen.height;
    const isAutoFit = ui.autoFitFrame;
    const effectiveSizeMode: SizeMode = isAutoFit ? img2img.sizeMode : "fixed";
    const genSize = resolveGenerationSize(effectiveSizeMode, frameW, frameH, img2img.scaleFactor, img2img.megapixelTarget);

    request.width_before = genSize.width;
    request.height_before = genSize.height;
    request.input_type = 1;

    // Flatten all image layers at full frame size
    const imageLayers = canvas.layers.filter((l) => l.type === "image") as ImageLayer[];
    const flattenedBlob = await flattenCanvas(imageLayers, frameW, frameH);
    if (flattenedBlob) {
      inputBlob = flattenedBlob;
      const ref = await uploadBlob(flattenedBlob, "input.png");
      request.inputs = [ref];
    }

    // Force resize_mode_before=1 (Fixed) + resize_name_before when scale/megapixel
    // so the backend resizes the init image to the computed target dimensions.
    // Both fields are required: run.py zeros resize_mode when resize_name is 'None'.
    if (effectiveSizeMode !== "fixed") {
      request.resize_mode_before = 1;
      request.resize_name_before = img2img.resizeMethod;
    }

    // Export mask from painted strokes if no explicit maskData
    let maskBlob: Blob | null = null;
    if (img2img.maskData) {
      // maskData is already a base64 string from external source — convert to Blob and upload
      const resp = await fetch(`data:image/png;base64,${img2img.maskData}`);
      maskBlob = await resp.blob();
    } else if (img2img.maskLines.length > 0) {
      maskBlob = await exportMask(img2img.maskLines, frameW, frameH);
    }
    if (maskBlob) {
      request.mask = await uploadBlob(maskBlob, "mask.png");
      request.mask_blur = img2img.maskBlur;
      request.inpaint_full_res = img2img.inpaintFullRes;
      request.inpaint_full_res_padding = img2img.inpaintFullResPadding;
      request.inpainting_mask_invert = img2img.inpaintingMaskInvert ? 1 : 0;
    }
  }

  // User override settings (merged last to take priority)
  if (Object.keys(gen.overrideSettings).length > 0) {
    request.extra = {
      ...request.extra,
      ...gen.overrideSettings,
    };
  }

  return { request, inputBlob };
}

/** Restore generation store state from a previous result. */
export function restoreFromResult(result: GenerationResult): void {
  const p = result.parameters;

  // Parse the info JSON for resolved values (actual seed used, etc.)
  let info: GenerationInfo | null = null;
  try { info = JSON.parse(result.info) as GenerationInfo; }
  catch { /* ignore */ }

  // Handle both control (extra) and legacy (override_settings) field names
  const overrides = (p.extra ?? p.override_settings ?? {}) as Record<string, unknown>;

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

    // Resolution — control uses width_before/height_before, legacy uses width/height
    width: num(p.width_before ?? p.width, 1024),
    height: num(p.height_before ?? p.height, 1024),

    // Batch — control uses batch_count, legacy uses n_iter
    batchSize: num(p.batch_size, 1),
    batchCount: num(p.batch_count ?? p.n_iter, 1),

    // Guidance — control uses pag_scale/pag_adaptive, legacy uses diffusers_ prefix
    cfgScale: num(p.cfg_scale, 7),
    cfgEnd: num(p.cfg_end, 1),
    guidanceRescale: num(p.diffusers_guidance_rescale, 0),
    imageCfgScale: num(p.image_cfg_scale, 6),
    pagScale: num(p.pag_scale ?? p.diffusers_pag_scale, 0),
    pagAdaptive: num(p.pag_adaptive ?? p.diffusers_pag_adaptive, 0.5),
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
    hiresResizeContext: str(p.hr_resize_context, "None"),

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

    // Generation modifiers (hijack)
    freeuEnabled: bool(p.freeu_enabled, false),
    freeuB1: num(p.freeu_b1, 1.2),
    freeuB2: num(p.freeu_b2, 1.4),
    freeuS1: num(p.freeu_s1, 0.9),
    freeuS2: num(p.freeu_s2, 0.2),
    hypertileUnetEnabled: bool(p.hypertile_unet_enabled, false),
    hypertileHiresOnly: bool(p.hypertile_hires_only, false),
    hypertileUnetTile: num(p.hypertile_unet_tile, 0),
    hypertileUnetMinTile: num(p.hypertile_unet_min_tile, 0),
    hypertileUnetSwapSize: num(p.hypertile_unet_swap_size, 1),
    hypertileUnetDepth: num(p.hypertile_unet_depth, 0),
    hypertileVaeEnabled: bool(p.hypertile_vae_enabled, false),
    hypertileVaeTile: num(p.hypertile_vae_tile, 128),
    hypertileVaeSwapSize: num(p.hypertile_vae_swap_size, 1),
    teacacheEnabled: bool(p.teacache_enabled, false),
    teacacheThresh: num(p.teacache_thresh, 0.15),
    tokenMergingMethod: str(p.token_merging_method, "None"),
    tomeRatio: num(p.tome_ratio, 0.0),
    todoRatio: num(p.todo_ratio, 0.0),

    // Latent corrections
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
    hdrColorPicker: str(p.hdr_color_picker, "#000000"),
    hdrTintRatio: num(p.hdr_tint_ratio, 0),

    // Color grading
    gradingBrightness: num(p.grading_brightness, 0),
    gradingContrast: num(p.grading_contrast, 0),
    gradingSaturation: num(p.grading_saturation, 0),
    gradingHue: num(p.grading_hue, 0),
    gradingGamma: num(p.grading_gamma, 1.0),
    gradingSharpness: num(p.grading_sharpness, 0),
    gradingColorTemp: num(p.grading_color_temp, 6500),
    gradingShadows: num(p.grading_shadows, 0),
    gradingMidtones: num(p.grading_midtones, 0),
    gradingHighlights: num(p.grading_highlights, 0),
    gradingClaheClip: num(p.grading_clahe_clip, 0),
    gradingClaheGrid: num(p.grading_clahe_grid, 8),
    gradingShadowsTint: str(p.grading_shadows_tint, "#000000"),
    gradingHighlightsTint: str(p.grading_highlights_tint, "#ffffff"),
    gradingSplitToneBalance: num(p.grading_split_tone_balance, 0.5),
    gradingVignette: num(p.grading_vignette, 0),
    gradingGrain: num(p.grading_grain, 0),
    gradingLutFile: str(p.grading_lut_file, ""),
    gradingLutStrength: num(p.grading_lut_strength, 1.0),

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

    // Scheduler overrides (top-level in new API, overrides dict in legacy)
    sigmaMethod: str(p.schedulers_sigma ?? overrides.schedulers_sigma, "default"),
    timestepSpacing: str(p.schedulers_timestep_spacing ?? overrides.schedulers_timestep_spacing, "default"),
    betaSchedule: str(p.schedulers_beta_schedule ?? overrides.schedulers_beta_schedule, "default"),
    predictionMethod: str(p.schedulers_prediction_type ?? overrides.schedulers_prediction_type, "default"),
    flowShift: num(p.schedulers_shift ?? overrides.schedulers_shift, 3),
    baseShift: num(p.schedulers_base_shift ?? overrides.schedulers_base_shift, 0.5),
    maxShift: num(p.schedulers_max_shift ?? overrides.schedulers_max_shift, 1.15),
    sigmaAdjust: num(p.schedulers_sigma_adjust ?? overrides.schedulers_sigma_adjust, 1.0),
    sigmaAdjustStart: num(p.schedulers_sigma_adjust_min ?? overrides.schedulers_sigma_adjust_min, 0.2),
    sigmaAdjustEnd: num(p.schedulers_sigma_adjust_max ?? overrides.schedulers_sigma_adjust_max, 1.0),
    thresholding: bool(p.schedulers_use_thresholding ?? overrides.schedulers_use_thresholding, false),
    dynamic: bool(p.schedulers_dynamic_shift ?? overrides.schedulers_dynamic_shift, false),
    rescale: bool(p.schedulers_rescale_betas ?? overrides.schedulers_rescale_betas, false),
    lowOrder: bool(p.schedulers_use_loworder ?? overrides.schedulers_use_loworder, true),
    timestepsOverride: str(p.schedulers_timesteps ?? overrides.schedulers_timesteps, ""),
    timestepsPreset: "None",

    // Detailer overrides
    ...(p.detailer_enabled ? {
      detailerModels: (Array.isArray(p.detailer_models) ? p.detailer_models : Array.isArray(overrides.detailer_models) ? overrides.detailer_models : ["face-yolo8n"]) as string[],
      detailerMaxDetected: num(p.detailer_max ?? overrides.detailer_max, 2),
      detailerPadding: num(p.detailer_padding ?? overrides.detailer_padding, 20),
      detailerBlur: num(p.detailer_blur ?? overrides.detailer_blur, 10),
      detailerConfidence: num(p.detailer_conf ?? overrides.detailer_conf, 0.6),
      detailerIou: num(p.detailer_iou ?? overrides.detailer_iou, 0.5),
      detailerMinSize: num(p.detailer_min_size ?? overrides.detailer_min_size, 0.0),
      detailerMaxSize: num(p.detailer_max_size ?? overrides.detailer_max_size, 1.0),
      detailerRenoise: num(p.detailer_sigma_adjust ?? overrides.detailer_sigma_adjust, 1.0),
      detailerRenoiseEnd: num(p.detailer_sigma_adjust_max ?? overrides.detailer_sigma_adjust_max, 1.0),
    } : {}),
  });

  // Restore input image and mask if present (img2img history)
  if (result.inputImage) {
    const w = num(p.width_before ?? p.width, 1024);
    const h = num(p.height_before ?? p.height, 1024);
    useCanvasStore.getState().restoreImageLayer(result.inputImage, w, h);

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
