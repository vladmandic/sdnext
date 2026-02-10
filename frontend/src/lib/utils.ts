import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatBytes(bytes: number, decimals = 2): string {
  if (bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(decimals))} ${sizes[i]}`;
}

export function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${mins}m ${secs.toFixed(0)}s`;
}

/** Return "#000" or "#fff" for best contrast against a hex background color. */
export function contrastText(hex: string): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return (r * 0.299 + g * 0.587 + b * 0.114) > 128 ? "#000" : "#fff";
}

export function base64ToBlob(base64: string, mimeType = "image/png"): Blob {
  const byteCharacters = atob(base64);
  const byteNumbers = new Array(byteCharacters.length);
  for (let i = 0; i < byteCharacters.length; i++) {
    byteNumbers[i] = byteCharacters.charCodeAt(i);
  }
  return new Blob([new Uint8Array(byteNumbers)], { type: mimeType });
}

const objectUrlCache = new Map<string, string>();

export function base64ToObjectUrl(base64: string, mimeType = "image/png"): string {
  const cached = objectUrlCache.get(base64);
  if (cached) return cached;
  const url = URL.createObjectURL(base64ToBlob(base64, mimeType));
  objectUrlCache.set(base64, url);
  return url;
}

/** Download a base64 image as a file */
export function downloadBase64Image(base64: string, filename: string, mimeType = "image/png"): void {
  const blob = base64ToBlob(base64, mimeType);
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/** Generate a filename from generation info */
export function generateImageFilename(info: string, imageIndex: number): string {
  let seed = "unknown";
  let model = "image";
  try {
    const parsed = JSON.parse(info);
    if (parsed.seed) seed = String(parsed.seed);
    if (parsed.model) model = String(parsed.model).split("/").pop()?.split(".")[0] ?? "image";
  } catch { /* fallback */ }
  const timestamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
  return `${model}_${seed}_${imageIndex}_${timestamp}.png`;
}
