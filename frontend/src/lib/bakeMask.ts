import { findConnectedComponents } from "@/lib/connectedComponents";
import { blobToBase64 } from "@/lib/image";
import { useCanvasStore, type MaskObjectLayer } from "@/stores/canvasStore";
import { useGenerationStore } from "@/stores/generationStore";
import { useImg2ImgStore, type MaskLine } from "@/stores/img2imgStore";

/** Render new (uncommitted) mask strokes as white-on-transparent. */
function renderStrokes(ctx: CanvasRenderingContext2D, lines: MaskLine[]) {
  for (const line of lines) {
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.lineWidth = line.strokeWidth;

    if (line.tool === "brush") {
      ctx.globalCompositeOperation = "source-over";
      ctx.strokeStyle = "#fff";
    } else {
      ctx.globalCompositeOperation = "destination-out";
      ctx.strokeStyle = "#fff";
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
 * Colorize a white-on-transparent region canvas with the mask display color.
 * Returns a new canvas where white pixels are replaced by the mask color (RGB)
 * while preserving the original alpha.
 */
function colorizeRegion(region: HTMLCanvasElement, rgb: string): HTMLCanvasElement {
  const w = region.width;
  const h = region.height;
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d")!;

  // Parse "#rrggbb" to components
  const r = parseInt(rgb.slice(1, 3), 16);
  const g = parseInt(rgb.slice(3, 5), 16);
  const b = parseInt(rgb.slice(5, 7), 16);

  // Copy source to get alpha channel
  ctx.drawImage(region, 0, 0);
  const imgData = ctx.getImageData(0, 0, w, h);
  const data = imgData.data;

  for (let i = 0; i < data.length; i += 4) {
    if (data[i + 3] > 0) {
      data[i] = r;
      data[i + 1] = g;
      data[i + 2] = b;
      // alpha stays as-is
    }
  }

  ctx.putImageData(imgData, 0, 0);
  return canvas;
}

/** Convert an HTMLCanvasElement to a base64 PNG string. */
function canvasToBase64(canvas: HTMLCanvasElement): Promise<string> {
  return new Promise((resolve) => {
    canvas.toBlob((blob) => {
      if (!blob) { resolve(""); return; }
      blobToBase64(blob).then(resolve);
    }, "image/png");
  });
}

/**
 * Check whether two bounding boxes overlap (used to match old mask objects
 * to new regions for preserving lock/visible state).
 */
function rectsOverlap(
  a: { x: number; y: number; w: number; h: number },
  b: { x: number; y: number; w: number; h: number },
): boolean {
  return a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;
}

/**
 * Bake current mask strokes into mask objects.
 *
 * 1. Renders existing mask objects + new strokes onto a full-frame canvas
 * 2. Runs connected-component labeling to find separate regions
 * 3. Creates colored MaskObjectLayers for each region
 * 4. Preserves lock/visible from old objects via bbox overlap
 * 5. Replaces old mask layers and clears strokes
 */
export async function bakeMaskStrokes(): Promise<void> {
  const { width: frameW, height: frameH } = useGenerationStore.getState();
  if (frameW <= 0 || frameH <= 0) return;

  const maskLines = useImg2ImgStore.getState().maskLines;
  const oldMasks = useCanvasStore.getState().getMaskLayers();

  // Nothing to bake
  if (maskLines.length === 0 && oldMasks.length === 0) return;

  // 1. Composite everything onto a white-on-transparent canvas
  const canvas = document.createElement("canvas");
  canvas.width = frameW;
  canvas.height = frameH;
  const ctx = canvas.getContext("2d")!;

  // Render existing mask objects first (white-on-transparent)
  // We need to load each object's image synchronously, so we pre-load them
  if (oldMasks.length > 0) {
    const loadedImages = await Promise.all(
      oldMasks.map((m) => new Promise<{ mask: MaskObjectLayer; img: HTMLImageElement }>((resolve) => {
        const img = new Image();
        img.onload = () => resolve({ mask: m, img });
        img.onerror = () => resolve({ mask: m, img });
        img.src = m.imageData;
      })),
    );

    for (const { mask: m, img } of loadedImages) {
      ctx.save();
      ctx.translate(m.x, m.y);
      ctx.rotate(m.rotation);
      ctx.scale(m.scaleX, m.scaleY);
      ctx.drawImage(img, 0, 0, m.width, m.height);
      ctx.restore();
    }

    // Convert whatever color was drawn to white, preserving alpha
    const imgData = ctx.getImageData(0, 0, frameW, frameH);
    const data = imgData.data;
    for (let i = 0; i < data.length; i += 4) {
      if (data[i + 3] > 0) {
        data[i] = 255;
        data[i + 1] = 255;
        data[i + 2] = 255;
      }
    }
    ctx.putImageData(imgData, 0, 0);
  }

  // Render new strokes on top
  if (maskLines.length > 0) {
    renderStrokes(ctx, maskLines);
  }

  // 2. Connected-component labeling
  const fullImageData = ctx.getImageData(0, 0, frameW, frameH);
  const regions = findConnectedComponents(fullImageData);

  if (regions.length === 0) {
    // All mask data was erased
    useCanvasStore.getState().removeMaskLayers();
    useImg2ImgStore.setState({ maskLines: [] });
    return;
  }

  // 3. Create new MaskObjectLayers
  const maskColor = useCanvasStore.getState().maskColor;
  const rgb = maskColor.slice(0, 7);

  const newLayers: MaskObjectLayer[] = await Promise.all(
    regions.map(async (region) => {
      const coloredCanvas = colorizeRegion(region.canvas, rgb);
      const base64 = await canvasToBase64(coloredCanvas);
      const blob = await new Promise<Blob | null>((resolve) => coloredCanvas.toBlob(resolve, "image/png"));
      const objectUrl = blob ? URL.createObjectURL(blob) : "";

      // Match against old masks to preserve lock/visible state
      let locked = true;   // default: locked
      let visible = true;
      for (const old of oldMasks) {
        const oldRect = { x: old.x, y: old.y, w: old.width * old.scaleX, h: old.height * old.scaleY };
        const newRect = { x: region.x, y: region.y, w: region.width, h: region.height };
        if (rectsOverlap(oldRect, newRect)) {
          locked = old.locked;
          visible = old.visible;
          break;
        }
      }

      return {
        id: crypto.randomUUID(),
        type: "mask" as const,
        name: `Mask ${region.x},${region.y}`,
        visible,
        opacity: 1,
        locked,
        imageData: objectUrl,
        base64,
        x: region.x,
        y: region.y,
        width: region.width,
        height: region.height,
        scaleX: 1,
        scaleY: 1,
        rotation: 0,
      };
    }),
  );

  // 4. Replace old mask layers and clear strokes
  useCanvasStore.getState().replaceMaskLayers(newLayers);
  useImg2ImgStore.setState({ maskLines: [] });
}
