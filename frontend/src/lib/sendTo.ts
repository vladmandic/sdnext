import { useVideoStore } from "@/stores/videoStore";
import { useVideoCanvasStore } from "@/stores/videoCanvasStore";
import { useProcessStore } from "@/stores/processStore";
import { useGenerationStore } from "@/stores/generationStore";
import { useCanvasStore } from "@/stores/canvasStore";
import { useUiStore } from "@/stores/uiStore";
import { fileToBase64 } from "@/lib/image";

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
  const store = useVideoStore.getState();
  const keyMap: Record<string, string> = {
    engine: "engine",
    model: "model",
    prompt: "prompt",
    negative: "negative",
    width: "width",
    height: "height",
    frames: "frames",
    steps: "steps",
    seed: "seed",
    guidance_scale: "guidanceScale",
    guidance_true: "guidanceTrue",
    sampler_shift: "samplerShift",
    dynamic_shift: "dynamicShift",
    init_strength: "initStrength",
    vae_type: "vaeType",
    vae_tile_frames: "vaeTileFrames",
    fps: "fps",
    interpolate: "interpolate",
    codec: "codec",
    format: "format",
  };
  const updates: Record<string, unknown> = {};
  for (const [apiKey, storeKey] of Object.entries(keyMap)) {
    if (apiKey in params && params[apiKey] !== undefined) {
      updates[storeKey] = params[apiKey];
    }
  }
  if (Object.keys(updates).length > 0) {
    store.setParams(updates);
  }
}
