import { useCanvasStore, type MaskObjectLayer } from "@/stores/canvasStore";
import type { MaskLine } from "@/stores/img2imgStore";

/**
 * Render mask objects as white-on-transparent onto an existing canvas context.
 * Each mask object's colored pixels are converted to white, preserving alpha
 * and respecting position/scale/rotation transforms.
 */
async function renderMaskObjects(ctx: CanvasRenderingContext2D, masks: MaskObjectLayer[]): Promise<void> {
  if (masks.length === 0) return;

  const loaded = await Promise.all(
    masks.filter((m) => m.visible).map((m) => new Promise<{ mask: MaskObjectLayer; img: HTMLImageElement }>((resolve) => {
      const img = new Image();
      img.onload = () => resolve({ mask: m, img });
      img.onerror = () => resolve({ mask: m, img });
      img.src = m.imageData;
    })),
  );

  // Draw each mask then convert non-transparent pixels to white
  const tmpCanvas = document.createElement("canvas");
  tmpCanvas.width = ctx.canvas.width;
  tmpCanvas.height = ctx.canvas.height;
  const tmpCtx = tmpCanvas.getContext("2d")!;

  for (const { mask: m, img } of loaded) {
    tmpCtx.clearRect(0, 0, tmpCanvas.width, tmpCanvas.height);
    tmpCtx.save();
    tmpCtx.translate(m.x, m.y);
    tmpCtx.rotate(m.rotation);
    tmpCtx.scale(m.scaleX, m.scaleY);
    tmpCtx.drawImage(img, 0, 0, m.width, m.height);
    tmpCtx.restore();
  }

  // Convert colored mask pixels to white
  const imgData = tmpCtx.getImageData(0, 0, tmpCanvas.width, tmpCanvas.height);
  const data = imgData.data;
  for (let i = 0; i < data.length; i += 4) {
    if (data[i + 3] > 0) {
      data[i] = 255;
      data[i + 1] = 255;
      data[i + 2] = 255;
      data[i + 3] = 255;
    }
  }
  tmpCtx.putImageData(imgData, 0, 0);

  // Composite white masks onto the export canvas
  ctx.drawImage(tmpCanvas, 0, 0);
}

/** Render mask strokes as white-on-black onto the canvas. */
function renderStrokes(ctx: CanvasRenderingContext2D, lines: MaskLine[]) {
  for (const line of lines) {
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.lineWidth = line.strokeWidth;

    if (line.tool === "brush") {
      ctx.globalCompositeOperation = "source-over";
      ctx.strokeStyle = "#fff";
    } else {
      ctx.globalCompositeOperation = "source-over";
      ctx.strokeStyle = "#000";
    }

    ctx.beginPath();
    const pts = line.points;
    if (pts.length >= 2) {
      ctx.moveTo(pts[0], pts[1]);
      for (let i = 2; i < pts.length; i += 2) {
        ctx.lineTo(pts[i], pts[i + 1]);
      }
    }
    ctx.stroke();
  }
}

/**
 * Export the complete mask (objects + any pending strokes) as a PNG Blob.
 * White = inpaint, black = keep.
 */
export async function exportMask(lines: MaskLine[], width: number, height: number): Promise<Blob | null> {
  if (width <= 0 || height <= 0) return null;

  const maskObjects = useCanvasStore.getState().getMaskLayers().filter((m) => m.visible);
  if (maskObjects.length === 0 && lines.length === 0) return null;

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;

  // Fill with black (keep)
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, width, height);

  // Render mask objects as white regions
  await renderMaskObjects(ctx, maskObjects);

  // Render any pending strokes (normally empty after baking)
  if (lines.length > 0) {
    renderStrokes(ctx, lines);
  }

  return new Promise<Blob | null>((resolve) => {
    canvas.toBlob((blob) => resolve(blob), "image/png");
  });
}
