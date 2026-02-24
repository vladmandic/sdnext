import type { ImageLayer } from "@/stores/canvasStore";
import type { FreeTransform, FitMode } from "@/lib/image";
import { computeFit } from "@/lib/image";

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}

/**
 * Composites all visible image layers within the frame bounds into a single
 * PNG Blob. Returns null if no visible layers exist.
 */
export async function flattenCanvas(
  layers: ImageLayer[],
  frameWidth: number,
  frameHeight: number,
): Promise<Blob | null> {
  const visible = layers.filter((l) => l.visible);
  if (visible.length === 0) return null;

  const canvas = document.createElement("canvas");
  canvas.width = frameWidth;
  canvas.height = frameHeight;
  const ctx = canvas.getContext("2d")!;

  for (const layer of visible) {
    const src = layer.base64 ? `data:image/png;base64,${layer.base64}` : layer.imageData;
    const img = await loadImage(src);
    ctx.save();
    ctx.translate(layer.x, layer.y);
    ctx.rotate((layer.rotation * Math.PI) / 180);
    ctx.scale(layer.scaleX, layer.scaleY);
    ctx.globalAlpha = layer.opacity;
    ctx.drawImage(img, 0, 0);
    ctx.restore();
  }

  return new Promise<Blob | null>((resolve) => {
    canvas.toBlob((blob) => resolve(blob), "image/png");
  });
}

/**
 * Composites an image onto a generation-sized canvas using the given fit mode.
 * WYSIWYG: the output matches what the user sees on the canvas frame.
 */
export async function compositeFitImage(
  file: File,
  genW: number,
  genH: number,
  fitMode: FitMode,
): Promise<Blob> {
  const url = URL.createObjectURL(file);
  const img = await loadImage(url);
  URL.revokeObjectURL(url);

  const canvas = document.createElement("canvas");
  canvas.width = genW;
  canvas.height = genH;
  const ctx = canvas.getContext("2d")!;

  const fit = computeFit(img.naturalWidth, img.naturalHeight, 0, 0, genW, genH, fitMode);
  if (fit.crop) {
    ctx.drawImage(img, fit.crop.x, fit.crop.y, fit.crop.width, fit.crop.height, fit.x, fit.y, fit.width, fit.height);
  } else {
    ctx.drawImage(img, fit.x, fit.y, fit.width, fit.height);
  }

  return new Promise<Blob>((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) resolve(blob);
      else reject(new Error("Failed to composite fit image"));
    }, "image/png");
  });
}

/**
 * Composites a free-mode control image onto a generation-sized canvas.
 * The transform is in display-unit space; displayScale converts to pixel space.
 */
export async function compositeControlImage(
  file: File,
  transform: FreeTransform,
  genW: number,
  genH: number,
  displayScale: number,
): Promise<Blob> {
  const url = URL.createObjectURL(file);
  const img = await loadImage(url);
  URL.revokeObjectURL(url);

  const canvas = document.createElement("canvas");
  canvas.width = genW;
  canvas.height = genH;
  const ctx = canvas.getContext("2d")!;

  // Convert display-unit transform to pixel space
  const pixelX = transform.x / displayScale;
  const pixelY = transform.y / displayScale;
  const pixelScaleX = transform.scaleX / displayScale;
  const pixelScaleY = transform.scaleY / displayScale;

  ctx.save();
  ctx.translate(pixelX, pixelY);
  ctx.rotate((transform.rotation * Math.PI) / 180);
  ctx.scale(pixelScaleX, pixelScaleY);
  ctx.drawImage(img, 0, 0);
  ctx.restore();

  return new Promise<Blob>((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) resolve(blob);
      else reject(new Error("Failed to composite control image"));
    }, "image/png");
  });
}
