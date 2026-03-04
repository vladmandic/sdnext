import type { ImagesSubTab } from "@/stores/uiStore";

export interface ParamEntry {
  tab: ImagesSubTab;
  section: string;
  param: string;
  label: string;
  keywords: string[];
}

export const PARAM_MAP: ParamEntry[] = [
  // --- Prompts ---
  { tab: "prompts", section: "size", param: "width", label: "Width", keywords: ["size", "resolution", "dimensions"] },
  { tab: "prompts", section: "size", param: "height", label: "Height", keywords: ["size", "resolution", "dimensions"] },
  { tab: "prompts", section: "batch", param: "count", label: "Batch Count", keywords: ["batch", "repeat", "iterations"] },
  { tab: "prompts", section: "batch", param: "size", label: "Batch Size", keywords: ["batch", "parallel", "simultaneous"] },

  // --- Sampler ---
  { tab: "sampler", section: "sampler", param: "method", label: "Sampler Method", keywords: ["sampler", "sampling", "euler", "dpm", "algorithm"] },
  { tab: "sampler", section: "sampler", param: "steps", label: "Steps", keywords: ["iterations", "quality", "sampling steps"] },
  { tab: "sampler", section: "scheduler", param: "sigma", label: "Sigma Method", keywords: ["scheduler", "karras", "exponential"] },
  { tab: "sampler", section: "scheduler", param: "spacing", label: "Timestep Spacing", keywords: ["scheduler", "linspace", "leading", "trailing"] },
  { tab: "sampler", section: "scheduler", param: "beta", label: "Beta Schedule", keywords: ["scheduler", "linear", "sigmoid"] },
  { tab: "sampler", section: "scheduler", param: "prediction", label: "Prediction Method", keywords: ["scheduler", "epsilon", "v_prediction", "flow"] },
  { tab: "sampler", section: "timesteps", param: "preset", label: "Timesteps Preset", keywords: ["timesteps"] },
  { tab: "sampler", section: "timesteps", param: "override", label: "Timesteps Override", keywords: ["timesteps", "custom"] },
  { tab: "sampler", section: "sigma", param: "start", label: "Sigma Start", keywords: ["sigma", "adjust"] },
  { tab: "sampler", section: "sigma", param: "end", label: "Sigma End", keywords: ["sigma", "adjust"] },
  { tab: "sampler", section: "sigma", param: "adjust", label: "Sigma Adjust", keywords: ["sigma"] },
  { tab: "sampler", section: "shifts", param: "flow shift", label: "Flow Shift", keywords: ["shift", "flow"] },
  { tab: "sampler", section: "shifts", param: "base shift", label: "Base Shift", keywords: ["shift", "base"] },
  { tab: "sampler", section: "shifts", param: "max shift", label: "Max Shift", keywords: ["shift", "maximum"] },
  { tab: "sampler", section: "seed", param: "seed", label: "Seed", keywords: ["random", "reproducible", "deterministic"] },
  { tab: "sampler", section: "seed", param: "variation", label: "Seed Variation", keywords: ["subseed", "variation", "var"] },
  { tab: "sampler", section: "seed", param: "var. str.", label: "Variation Strength", keywords: ["subseed", "variation strength"] },

  // --- Guidance ---
  { tab: "guidance", section: "guidance", param: "guidance scale", label: "Guidance Scale", keywords: ["cfg", "classifier free", "prompt adherence"] },
  { tab: "guidance", section: "guidance", param: "guidance end", label: "Guidance End", keywords: ["cfg", "end step"] },
  { tab: "guidance", section: "guidance", param: "rescale", label: "Guidance Rescale", keywords: ["cfg", "rescale"] },
  { tab: "guidance", section: "refine guidance", param: "refine guidance scale", label: "Refine Guidance Scale", keywords: ["cfg", "refine", "second pass"] },
  { tab: "guidance", section: "attention guidance", param: "pag scale", label: "PAG Scale", keywords: ["pag", "perturbed attention", "guidance"] },
  { tab: "guidance", section: "attention guidance", param: "adaptive", label: "PAG Adaptive", keywords: ["pag", "adaptive", "scaling"] },

  // --- Refine ---
  { tab: "refine", section: "hires fix", param: "scale", label: "Hires Scale", keywords: ["hires", "upscale", "highres", "resolution"] },
  { tab: "refine", section: "hires fix", param: "denoise", label: "Hires Denoise", keywords: ["hires", "denoising", "strength"] },
  { tab: "refine", section: "hires fix", param: "steps", label: "Hires Steps", keywords: ["hires", "steps", "highres"] },
  { tab: "refine", section: "hires fix", param: "mode", label: "Hires Mode", keywords: ["hires", "resize", "upscale mode"] },
  { tab: "refine", section: "hires fix", param: "upscaler", label: "Hires Upscaler", keywords: ["hires", "upscaler", "model"] },
  { tab: "refine", section: "hires fix", param: "sampler", label: "Hires Sampler", keywords: ["hires", "sampler", "second pass"] },
  { tab: "refine", section: "refiner", param: "start", label: "Refiner Start", keywords: ["refiner", "switch", "handoff"] },

  // --- Detail ---
  { tab: "detail", section: "detailer", param: "models", label: "Detailer Models", keywords: ["detailer", "model", "detection"] },
  { tab: "detail", section: "generation", param: "steps", label: "Detailer Steps", keywords: ["detailer", "steps", "inpaint"] },
  { tab: "detail", section: "generation", param: "strength", label: "Detailer Strength", keywords: ["detailer", "denoising", "strength"] },
  { tab: "detail", section: "generation", param: "resolution", label: "Detailer Resolution", keywords: ["detailer", "resolution", "size"] },
  { tab: "detail", section: "detection", param: "confidence", label: "Detection Confidence", keywords: ["detailer", "threshold", "confidence"] },
  { tab: "detail", section: "detection", param: "iou", label: "Detection IoU", keywords: ["detailer", "overlap", "iou"] },
  { tab: "detail", section: "detection", param: "min size", label: "Detection Min Size", keywords: ["detailer", "minimum", "size"] },
  { tab: "detail", section: "detection", param: "max size", label: "Detection Max Size", keywords: ["detailer", "maximum", "size"] },
  { tab: "detail", section: "detection", param: "classes", label: "Detection Classes", keywords: ["detailer", "classes", "person", "face"] },
  { tab: "detail", section: "options", param: "padding", label: "Detailer Padding", keywords: ["detailer", "padding", "margin"] },
  { tab: "detail", section: "options", param: "blur", label: "Detailer Blur", keywords: ["detailer", "blur", "mask"] },
  { tab: "detail", section: "options", param: "max detect", label: "Max Detections", keywords: ["detailer", "limit", "count"] },
  { tab: "detail", section: "noise", param: "renoise", label: "Renoise", keywords: ["detailer", "noise", "renoise"] },

  // --- Advanced ---
  { tab: "advanced", section: "advanced", param: "clip skip", label: "CLIP Skip", keywords: ["clip", "skip", "layers", "text encoder"] },
  { tab: "advanced", section: "advanced", param: "vae type", label: "VAE Type", keywords: ["vae", "decoder", "encoder"] },

  // --- Color ---
  { tab: "color", section: "color correction", param: "method", label: "Color Correction Method", keywords: ["color", "correction", "histogram", "wavelet"] },
  { tab: "color", section: "latent corrections", param: "brightness", label: "Latent Brightness", keywords: ["latent", "brightness", "hdr"] },
  { tab: "color", section: "latent corrections", param: "sharpen", label: "Latent Sharpen", keywords: ["latent", "sharpen", "hdr"] },
  { tab: "color", section: "latent corrections", param: "color", label: "Latent Color", keywords: ["latent", "color", "saturation"] },
  { tab: "color", section: "latent corrections", param: "range", label: "Latent Clamp Range", keywords: ["latent", "clamp", "hdr"] },
  { tab: "color", section: "latent corrections", param: "threshold", label: "Latent Clamp Threshold", keywords: ["latent", "clamp", "threshold"] },
  { tab: "color", section: "latent corrections", param: "center", label: "HDR Maximize Center", keywords: ["hdr", "maximize", "center"] },
  { tab: "color", section: "latent corrections", param: "max range", label: "HDR Maximize Range", keywords: ["hdr", "maximize", "range"] },
  { tab: "color", section: "latent corrections", param: "tint strength", label: "Tint Strength", keywords: ["tint", "color", "latent"] },
  { tab: "color", section: "basic", param: "brightness", label: "Grading Brightness", keywords: ["grading", "brightness", "exposure"] },
  { tab: "color", section: "basic", param: "contrast", label: "Contrast", keywords: ["grading", "contrast"] },
  { tab: "color", section: "basic", param: "saturation", label: "Saturation", keywords: ["grading", "saturation", "vibrance"] },
  { tab: "color", section: "basic", param: "hue", label: "Hue", keywords: ["grading", "hue", "color shift"] },
  { tab: "color", section: "basic", param: "gamma", label: "Gamma", keywords: ["grading", "gamma", "midtones"] },
  { tab: "color", section: "basic", param: "sharpness", label: "Sharpness", keywords: ["grading", "sharpen", "detail"] },
  { tab: "color", section: "basic", param: "color temp (k)", label: "Color Temperature", keywords: ["grading", "temperature", "kelvin", "warmth"] },
  { tab: "color", section: "tone", param: "shadows", label: "Shadows", keywords: ["grading", "shadows", "dark"] },
  { tab: "color", section: "tone", param: "midtones", label: "Midtones", keywords: ["grading", "midtones", "tone"] },
  { tab: "color", section: "tone", param: "highlights", label: "Highlights", keywords: ["grading", "highlights", "bright"] },
  { tab: "color", section: "tone", param: "clahe clip", label: "CLAHE Clip", keywords: ["clahe", "adaptive", "histogram"] },
  { tab: "color", section: "tone", param: "clahe grid", label: "CLAHE Grid", keywords: ["clahe", "grid", "tiles"] },
  { tab: "color", section: "split toning", param: "balance", label: "Split Tone Balance", keywords: ["split", "toning", "balance"] },
  { tab: "color", section: "effects", param: "vignette", label: "Vignette", keywords: ["grading", "vignette", "edges"] },
  { tab: "color", section: "effects", param: "grain", label: "Grain", keywords: ["grading", "grain", "noise", "film"] },
  { tab: "color", section: "lut", param: "strength", label: "LUT Strength", keywords: ["lut", "lookup", "color grading"] },
];
