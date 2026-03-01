import { useVideoStore } from "@/stores/videoStore";
import { useVideoCanvasStore } from "@/stores/videoCanvasStore";
import { useProcessStore } from "@/stores/processStore";
import { useGenerationStore } from "@/stores/generationStore";
import { useCanvasStore } from "@/stores/canvasStore";
import { useUiStore } from "@/stores/uiStore";
import { fileToBase64, base64ToFile } from "@/lib/image";
import { resolveImageSrc } from "@/lib/utils";
import type { DragPayload } from "@/stores/dragStore";

export function extractFrameFromVideo(videoUrl: string, time: number): Promise<Blob> {
  return new Promise((resolve, reject) => {
    const video = document.createElement("video");
    video.crossOrigin = "anonymous";
    video.preload = "auto";
    video.muted = true;

    const cleanup = () => {
      video.removeAttribute("src");
      video.load();
    };

    video.addEventListener("error", () => {
      cleanup();
      reject(new Error("Failed to load video"));
    });

    video.addEventListener("loadedmetadata", () => {
      const clampedTime = Math.min(Math.max(0, time), video.duration);
      video.currentTime = clampedTime;
    });

    video.addEventListener("seeked", () => {
      const canvas = document.createElement("canvas");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        cleanup();
        reject(new Error("Canvas 2D context unavailable"));
        return;
      }
      ctx.drawImage(video, 0, 0);
      canvas.toBlob((blob) => {
        cleanup();
        if (blob) resolve(blob);
        else reject(new Error("Failed to capture frame"));
      }, "image/png");
    }, { once: true });

    video.src = videoUrl;
  });
}

export function blobToFile(blob: Blob, filename = "frame.png"): File {
  return new File([blob], filename, { type: blob.type });
}

export async function sendFrameToVideoInit(blob: Blob) {
  const file = blobToFile(blob, "init-frame.png");
  const base64 = await fileToBase64(file);
  const objectUrl = URL.createObjectURL(file);
  const img = new window.Image();
  img.src = objectUrl;
  await new Promise<void>((r) => { img.onload = () => r(); });
  useVideoCanvasStore.getState().setFrame("init", file, base64, objectUrl, img.naturalWidth, img.naturalHeight);
  useUiStore.getState().setSidebarView("video");
}

export async function sendFrameToVideoLast(blob: Blob) {
  const file = blobToFile(blob, "last-frame.png");
  const base64 = await fileToBase64(file);
  const objectUrl = URL.createObjectURL(file);
  const img = new window.Image();
  img.src = objectUrl;
  await new Promise<void>((r) => { img.onload = () => r(); });
  useVideoCanvasStore.getState().setFrame("last", file, base64, objectUrl, img.naturalWidth, img.naturalHeight);
  useUiStore.getState().setSidebarView("video");
}

export function sendFrameToUpscale(blob: Blob) {
  useProcessStore.getState().setImage(blobToFile(blob, "frame.png"));
  useUiStore.getState().setSidebarView("process");
}

export async function fetchRemoteImage(url: string, filename = "image.png"): Promise<File> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch image: ${res.status}`);
  const blob = await res.blob();
  return blobToFile(blob, filename);
}

export async function sendImageToCanvas(file: File) {
  const base64 = await fileToBase64(file);
  const objectUrl = URL.createObjectURL(file);
  const img = new window.Image();
  img.src = objectUrl;
  await new Promise<void>((r) => { img.onload = () => r(); });
  useCanvasStore.getState().addImageLayer(file, base64, objectUrl, img.naturalWidth, img.naturalHeight);
  useUiStore.getState().setSidebarView("images");
}

export function sendPromptToGeneration(prompt: string, negative?: string) {
  const gen = useGenerationStore.getState();
  gen.setParam("prompt", prompt);
  if (negative) gen.setParam("negativePrompt", negative);
  useUiStore.getState().setSidebarView("images");
}

export function sendPromptToVideo(prompt: string, negative?: string) {
  const store = useVideoStore.getState();
  store.setParam("prompt", prompt);
  if (negative) store.setParam("negative", negative);
  useUiStore.getState().setSidebarView("video");
}

export function appendToGenerationPrompt(text: string) {
  const gen = useGenerationStore.getState();
  const current = gen.prompt;
  gen.setParam("prompt", current ? `${current} ${text}` : text);
}

export function restoreVideoSettings(params: Record<string, unknown>) {
  const num = (v: unknown, fallback?: number) => (typeof v === "number" ? v : fallback);
  const str = (v: unknown, fallback?: string) => (typeof v === "string" ? v : fallback);
  const bool = (v: unknown, fallback?: boolean) => (typeof v === "boolean" ? v : fallback);

  const sharedKeyMap: Record<string, (p: Record<string, unknown>) => unknown> = {
    engine: (p) => str(p.engine),
    model: (p) => str(p.model),
    prompt: (p) => str(p.prompt),
    negative: (p) => str(p.negative),
    width: (p) => num(p.width),
    height: (p) => num(p.height),
    frames: (p) => num(p.frames),
    steps: (p) => num(p.steps),
    seed: (p) => num(p.seed),
    guidanceScale: (p) => num(p.guidance_scale),
    guidanceTrue: (p) => num(p.guidance_true),
    sampler: (p) => num(p.sampler),
    samplerShift: (p) => num(p.sampler_shift),
    dynamicShift: (p) => bool(p.dynamic_shift),
    initStrength: (p) => num(p.init_strength),
    vaeType: (p) => str(p.vae_type),
    vaeTileFrames: (p) => num(p.vae_tile_frames),
  };

  const outputKeyMap: Record<string, (p: Record<string, unknown>) => unknown> = {
    fps: (p) => num(p.fps),
    interpolate: (p) => num(p.interpolate),
    codec: (p) => str(p.codec),
    format: (p) => str(p.format),
    codecOptions: (p) => str(p.codec_options),
    saveVideo: (p) => bool(p.save_video),
    saveFrames: (p) => bool(p.save_frames),
    saveSafetensors: (p) => bool(p.save_safetensors),
  };

  const fpKeyMap: Record<string, (p: Record<string, unknown>) => unknown> = {
    fpVariant: (p) => str(p.fp_variant),
    fpResolution: (p) => num(p.fp_resolution),
    fpDuration: (p) => num(p.fp_duration),
    fpLatentWindowSize: (p) => num(p.fp_latent_window_size),
    fpSteps: (p) => num(p.fp_steps),
    fpShift: (p) => num(p.fp_shift),
    fpCfgScale: (p) => num(p.fp_cfg_scale),
    fpCfgDistilled: (p) => num(p.fp_cfg_distilled),
    fpCfgRescale: (p) => num(p.fp_cfg_rescale),
    fpStartWeight: (p) => num(p.fp_start_weight),
    fpEndWeight: (p) => num(p.fp_end_weight),
    fpVisionWeight: (p) => num(p.fp_vision_weight),
    fpSectionPrompt: (p) => str(p.fp_section_prompt),
    fpSystemPrompt: (p) => str(p.fp_system_prompt),
    fpTeacache: (p) => bool(p.fp_teacache),
    fpOptimizedPrompt: (p) => bool(p.fp_optimized_prompt),
    fpCfgZero: (p) => bool(p.fp_cfg_zero),
    fpPreview: (p) => bool(p.fp_preview),
    fpAttention: (p) => str(p.fp_attention),
    fpVaeType: (p) => str(p.fp_vae_type),
  };

  const ltxKeyMap: Record<string, (p: Record<string, unknown>) => unknown> = {
    ltxModel: (p) => str(p.ltx_model),
    ltxSteps: (p) => num(p.ltx_steps),
    ltxDecodeTimestep: (p) => num(p.ltx_decode_timestep),
    ltxNoiseScale: (p) => num(p.ltx_noise_scale),
    ltxUpsampleEnable: (p) => bool(p.ltx_upsample_enable),
    ltxUpsampleRatio: (p) => num(p.ltx_upsample_ratio),
    ltxRefineEnable: (p) => bool(p.ltx_refine_enable),
    ltxRefineStrength: (p) => num(p.ltx_refine_strength),
    ltxConditionStrength: (p) => num(p.ltx_condition_strength),
    ltxAudioEnable: (p) => bool(p.ltx_audio_enable),
  };

  const domain = str(params.domain ?? params.type, "") as string;
  const maps: Record<string, (p: Record<string, unknown>) => unknown>[] = [sharedKeyMap, outputKeyMap];
  if (domain === "framepack" || domain === "") maps.push(fpKeyMap);
  if (domain === "ltx" || domain === "") maps.push(ltxKeyMap);

  const updates: Record<string, unknown> = {};
  for (const map of maps) {
    for (const [storeKey, extractor] of Object.entries(map)) {
      const value = extractor(params);
      if (value !== undefined) updates[storeKey] = value;
    }
  }

  if (Object.keys(updates).length > 0) {
    useVideoStore.getState().setParams(updates);
  }
}

export async function sendResultToCanvas(result: { images: string[] }, imageIndex: number): Promise<void> {
  const raw = result.images[imageIndex];
  const src = resolveImageSrc(raw);
  const file = await fetchRemoteImage(src, "result.png");
  await sendImageToCanvas(file);
}

export async function sendResultToUpscale(result: { images: string[] }, imageIndex: number): Promise<void> {
  const raw = result.images[imageIndex];
  const src = resolveImageSrc(raw);
  const res = await fetch(src);
  const blob = await res.blob();
  sendFrameToUpscale(blob);
}

export async function payloadToFile(payload: DragPayload): Promise<File> {
  if (payload.type === "result-image" && payload.resultId != null && payload.imageIndex != null) {
    const result = useGenerationStore.getState().results.find((r) => r.id === payload.resultId);
    if (result) {
      const raw = result.images[payload.imageIndex];
      if (raw) {
        const src = resolveImageSrc(raw);
        if (src.startsWith("data:") || src.startsWith("blob:") || src.startsWith("/") || src.startsWith("http")) {
          return fetchRemoteImage(src, "result.png");
        }
        // Raw base64 (no data: prefix)
        return base64ToFile(raw, "result.png");
      }
    }
  }

  if (payload.type === "gallery-image" && payload.filePath) {
    return fetchRemoteImage(`/file=${payload.filePath}`, payload.filePath.split("/").pop() ?? "gallery.png");
  }

  throw new Error("Cannot resolve drag payload to file");
}
