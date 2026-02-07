import type { MaskLine } from "@/stores/img2imgStore";

/**
 * Renders mask lines to a base64 PNG string (white = inpaint, black = keep).
 * Returns null if no lines are provided.
 */
export function exportMaskToBase64(lines: MaskLine[], width: number, height: number): string | null {
  if (lines.length === 0 || width <= 0 || height <= 0) return null;

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;

  // Fill with black (keep)
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, width, height);

  for (const line of lines) {
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.lineWidth = line.strokeWidth;

    if (line.tool === "brush") {
      // White = inpaint area
      ctx.globalCompositeOperation = "source-over";
      ctx.strokeStyle = "#fff";
    } else {
      // Eraser: paint black back over white
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

  return canvas.toDataURL("image/png").split(",")[1];
}
