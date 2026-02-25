import resize from "@jsquash/resize";

export type ResizeMethod = "lanczos3" | "magicKernelSharp2021" | "mitchell" | "catrom" | "triangle";

const DEFAULT_METHOD: ResizeMethod = "magicKernelSharp2021";

// iOS Safari caps canvas at ~16.7MP; other mobile browsers have similar limits.
// Leave headroom below the hard limit to account for multiple concurrent canvases.
const MAX_CANVAS_PIXELS = 16_000_000;

// ─── Capability detection ───────────────────────────────────────────────────
//
// On constrained devices (mobile, low-memory) we fall back from WASM MKS2021
// to createImageBitmap with resizeQuality:"high" (browser-native bicubic).
// Still much better than the raw bilinear drawImage that was used before.

let _constrained: boolean | null = null;

function isConstrained(): boolean {
  if (_constrained !== null) return _constrained;
  const ua = navigator.userAgent;
  const isMobile = /Android|iPhone|iPad|iPod|Mobile/i.test(ua);
  // navigator.deviceMemory is Chrome-only (GB), absent on Safari/Firefox
  const lowMemory = "deviceMemory" in navigator && (navigator as { deviceMemory: number }).deviceMemory < 4;
  _constrained = isMobile || lowMemory;
  return _constrained;
}

/** Check whether a canvas at w×h would exceed the platform's pixel limit. */
export function canvasFitsLimit(w: number, h: number): boolean {
  return w * h <= MAX_CANVAS_PIXELS;
}

// ─── Fast path: browser-native resize via createImageBitmap ─────────────────

function canvasToBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise<Blob>((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) resolve(blob);
      else reject(new Error("canvasToBlob: toBlob returned null"));
    }, "image/png");
  });
}

async function nativeResize(canvas: HTMLCanvasElement, targetW: number, targetH: number): Promise<Blob> {
  const blob = await canvasToBlob(canvas);
  const bitmap = await createImageBitmap(blob, { resizeWidth: targetW, resizeHeight: targetH, resizeQuality: "high" });
  const out = document.createElement("canvas");
  out.width = targetW;
  out.height = targetH;
  out.getContext("2d")!.drawImage(bitmap, 0, 0);
  bitmap.close();
  return canvasToBlob(out);
}

// ─── WASM path: @jsquash/resize with MKS2021 ───────────────────────────────

function canvasToImageData(canvas: HTMLCanvasElement): ImageData {
  return canvas.getContext("2d")!.getImageData(0, 0, canvas.width, canvas.height);
}

function imageDataToBlob(imageData: ImageData): Promise<Blob> {
  const canvas = document.createElement("canvas");
  canvas.width = imageData.width;
  canvas.height = imageData.height;
  canvas.getContext("2d")!.putImageData(imageData, 0, 0);
  return canvasToBlob(canvas);
}

async function wasmResize(canvas: HTMLCanvasElement, targetW: number, targetH: number, method: ResizeMethod): Promise<Blob> {
  const sourceData = canvasToImageData(canvas);
  const resized = await resize(sourceData, {
    width: targetW,
    height: targetH,
    method,
    linearRGB: true,
    premultiply: true,
  });
  return imageDataToBlob(resized);
}

// ─── Public API ─────────────────────────────────────────────────────────────

/**
 * High-quality resize of a canvas element to target dimensions.
 *
 * Desktop: WASM MKS2021 (same algorithm family as backend Sharpfin),
 *          with sRGB linearization and alpha premultiplication.
 * Mobile/constrained: createImageBitmap with resizeQuality:"high"
 *          (browser-native bicubic — still far better than raw bilinear).
 *
 * Returns a PNG Blob at exactly targetW × targetH.
 * If the canvas is already at the target size, exports directly.
 */
export async function resizeCanvas(
  canvas: HTMLCanvasElement,
  targetW: number,
  targetH: number,
  method: ResizeMethod = DEFAULT_METHOD,
): Promise<Blob> {
  if (canvas.width === targetW && canvas.height === targetH) {
    return canvasToBlob(canvas);
  }
  if (isConstrained()) {
    return nativeResize(canvas, targetW, targetH);
  }
  return wasmResize(canvas, targetW, targetH, method);
}

/**
 * High-quality resize of a Blob (PNG/JPEG image) to target dimensions.
 * Decodes via createImageBitmap, draws to canvas, then resizes.
 */
export async function resizeBlob(
  blob: Blob,
  targetW: number,
  targetH: number,
  method: ResizeMethod = DEFAULT_METHOD,
): Promise<Blob> {
  const bitmap = await createImageBitmap(blob);
  if (bitmap.width === targetW && bitmap.height === targetH) {
    bitmap.close();
    return blob;
  }
  const canvas = document.createElement("canvas");
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  canvas.getContext("2d")!.drawImage(bitmap, 0, 0);
  bitmap.close();
  return resizeCanvas(canvas, targetW, targetH, method);
}
