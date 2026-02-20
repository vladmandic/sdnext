import type { ImageLayer } from "@/stores/canvasStore";

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
