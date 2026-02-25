import type { ImageLayer } from "@/stores/canvasStore";
import type { FreeTransform, FitMode } from "@/lib/image";
import { computeFit } from "@/lib/image";
import { resizeCanvas, canvasFitsLimit } from "@/lib/resize";

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
 * PNG Blob. When the composite involves significant downscaling, an internal
 * 2× oversampled canvas is used so that Canvas 2D's bilinear interpolation
 * operates on more source data, then the final resize to frameWidth×frameHeight
 * is performed via WASM MKS2021 for maximum quality.
 */
export async function flattenCanvas(
  layers: ImageLayer[],
  frameWidth: number,
  frameHeight: number,
): Promise<Blob | null> {
  const visible = layers.filter((l) => l.visible);
  if (visible.length === 0) return null;

  // Determine if any layer is being downscaled significantly (>25% reduction).
  // If so, composite at 2× the target to give bilinear more source pixels,
  // then let MKS2021 handle the final high-quality downsample.
  let needsHQResize = false;
  for (const layer of visible) {
    const drawnW = Math.abs(layer.scaleX) * frameWidth;
    const drawnH = Math.abs(layer.scaleY) * frameHeight;
    if (drawnW > frameWidth * 1.25 || drawnH > frameHeight * 1.25) {
      needsHQResize = true;
      break;
    }
  }

  // Only oversample if the 2× canvas fits within the platform's pixel limit
  // (iOS Safari caps at ~16.7MP, so a 2× of 2048×2048 is right at the edge)
  const canOversample = needsHQResize && canvasFitsLimit(frameWidth * 2, frameHeight * 2);
  const scale = canOversample ? 2 : 1;
  const internalW = frameWidth * scale;
  const internalH = frameHeight * scale;

  const canvas = document.createElement("canvas");
  canvas.width = internalW;
  canvas.height = internalH;
  const ctx = canvas.getContext("2d")!;
  ctx.imageSmoothingQuality = "high";

  for (const layer of visible) {
    const src = layer.base64 ? `data:image/png;base64,${layer.base64}` : layer.imageData;
    const img = await loadImage(src);
    ctx.save();
    ctx.translate(layer.x * scale, layer.y * scale);
    ctx.rotate((layer.rotation * Math.PI) / 180);
    ctx.scale(layer.scaleX * scale, layer.scaleY * scale);
    ctx.globalAlpha = layer.opacity;
    ctx.drawImage(img, 0, 0);
    ctx.restore();
  }

  // Final high-quality resize from internal resolution to target frame size
  return resizeCanvas(canvas, frameWidth, frameHeight);
}

/**
 * Composites an image onto a generation-sized canvas using the given fit mode.
 *
 * When the source image fits in a canvas (desktop), it is pre-resized via WASM
 * MKS2021 so Canvas 2D only does a 1:1 blit. On constrained devices where the
 * source is too large for an intermediate canvas, falls back to Canvas 2D with
 * imageSmoothingQuality:"high" (browser-native bicubic).
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

  const fit = computeFit(img.naturalWidth, img.naturalHeight, 0, 0, genW, genH, fitMode);
  const srcW = fit.crop ? Math.round(fit.crop.width) : img.naturalWidth;
  const srcH = fit.crop ? Math.round(fit.crop.height) : img.naturalHeight;

  // If the source image fits within the canvas pixel limit, use the WASM path:
  // extract source region at native resolution, resize via MKS2021, 1:1 blit.
  if (canvasFitsLimit(srcW, srcH)) {
    const srcCanvas = document.createElement("canvas");
    if (fit.crop) {
      srcCanvas.width = srcW;
      srcCanvas.height = srcH;
      srcCanvas.getContext("2d")!.drawImage(img, -fit.crop.x, -fit.crop.y);
    } else {
      srcCanvas.width = srcW;
      srcCanvas.height = srcH;
      srcCanvas.getContext("2d")!.drawImage(img, 0, 0);
    }

    const drawW = Math.round(fit.width);
    const drawH = Math.round(fit.height);
    const resizedBlob = await resizeCanvas(srcCanvas, drawW, drawH);
    const resizedBitmap = await createImageBitmap(resizedBlob);

    const outCanvas = document.createElement("canvas");
    outCanvas.width = genW;
    outCanvas.height = genH;
    outCanvas.getContext("2d")!.drawImage(resizedBitmap, Math.round(fit.x), Math.round(fit.y));
    resizedBitmap.close();

    return new Promise<Blob>((resolve, reject) => {
      outCanvas.toBlob((blob) => {
        if (blob) resolve(blob);
        else reject(new Error("Failed to composite fit image"));
      }, "image/png");
    });
  }

  // Fallback: let Canvas 2D handle the downscale with its best interpolation
  const canvas = document.createElement("canvas");
  canvas.width = genW;
  canvas.height = genH;
  const ctx = canvas.getContext("2d")!;
  ctx.imageSmoothingQuality = "high";
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
 *
 * When the image is being downscaled and the source fits within the canvas
 * pixel limit, it is pre-resized via WASM MKS2021 so Canvas 2D only handles
 * rotation/positioning at near-1:1 scale. Falls back to Canvas 2D with
 * imageSmoothingQuality:"high" on constrained devices.
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

  // Convert display-unit transform to pixel space
  const pixelX = transform.x / displayScale;
  const pixelY = transform.y / displayScale;
  const pixelScaleX = transform.scaleX / displayScale;
  const pixelScaleY = transform.scaleY / displayScale;

  // Compute the drawn pixel dimensions
  const drawnW = Math.abs(img.naturalWidth * pixelScaleX);
  const drawnH = Math.abs(img.naturalHeight * pixelScaleY);

  const canvas = document.createElement("canvas");
  canvas.width = genW;
  canvas.height = genH;
  const ctx = canvas.getContext("2d")!;
  ctx.imageSmoothingQuality = "high";

  // Use WASM pre-resize when downscaling significantly and source fits in a canvas
  const isDownscaling = drawnW > 0 && drawnH > 0 && (drawnW < img.naturalWidth * 0.9 || drawnH < img.naturalHeight * 0.9);
  const canPreResize = isDownscaling && canvasFitsLimit(img.naturalWidth, img.naturalHeight);

  if (canPreResize) {
    const srcCanvas = document.createElement("canvas");
    srcCanvas.width = img.naturalWidth;
    srcCanvas.height = img.naturalHeight;
    srcCanvas.getContext("2d")!.drawImage(img, 0, 0);
    const resizedBlob = await resizeCanvas(srcCanvas, Math.round(drawnW), Math.round(drawnH));
    const resizedBitmap = await createImageBitmap(resizedBlob);

    const flipX = Math.sign(pixelScaleX) || 1;
    const flipY = Math.sign(pixelScaleY) || 1;
    ctx.save();
    ctx.translate(pixelX, pixelY);
    ctx.rotate((transform.rotation * Math.PI) / 180);
    ctx.scale(flipX, flipY);
    ctx.drawImage(resizedBitmap, 0, 0);
    ctx.restore();
    resizedBitmap.close();
  } else {
    ctx.save();
    ctx.translate(pixelX, pixelY);
    ctx.rotate((transform.rotation * Math.PI) / 180);
    ctx.scale(pixelScaleX, pixelScaleY);
    ctx.drawImage(img, 0, 0);
    ctx.restore();
  }

  return new Promise<Blob>((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) resolve(blob);
      else reject(new Error("Failed to composite control image"));
    }, "image/png");
  });
}
