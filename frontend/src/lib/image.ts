export function base64ToImage(base64: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = base64.startsWith("data:") ? base64 : `data:image/png;base64,${base64}`;
  });
}

export function imageToBase64(canvas: HTMLCanvasElement, mimeType = "image/png"): string {
  return canvas.toDataURL(mimeType).split(",")[1];
}

export function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve((reader.result as string).split(",")[1]);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

export function base64ToFile(base64: string, name: string, mimeType = "image/png"): File {
  const byteChars = atob(base64);
  const bytes = new Uint8Array(byteChars.length);
  for (let i = 0; i < byteChars.length; i++) bytes[i] = byteChars.charCodeAt(i);
  return new File([bytes], name, { type: mimeType });
}

export type FitMode = "contain" | "cover" | "fill";

export interface FitResult {
  x: number;
  y: number;
  width: number;
  height: number;
  crop: { x: number; y: number; width: number; height: number } | null;
}

/** Compute position, size, and optional source crop for fitting an image into a frame. */
export function computeFit(imgW: number, imgH: number, frameX: number, frameY: number, frameW: number, frameH: number, mode: FitMode): FitResult {
  if (mode === "fill") {
    return { x: frameX, y: frameY, width: frameW, height: frameH, crop: null };
  }
  if (mode === "contain") {
    const scale = Math.min(frameW / imgW, frameH / imgH);
    const w = imgW * scale;
    const h = imgH * scale;
    return { x: frameX + (frameW - w) / 2, y: frameY + (frameH - h) / 2, width: w, height: h, crop: null };
  }
  // cover: scale up so image fills frame, crop the overflow in source space
  const scale = Math.max(frameW / imgW, frameH / imgH);
  const visibleW = frameW / scale;
  const visibleH = frameH / scale;
  return {
    x: frameX, y: frameY, width: frameW, height: frameH,
    crop: { x: (imgW - visibleW) / 2, y: (imgH - visibleH) / 2, width: visibleW, height: visibleH },
  };
}

export function createObjectUrl(base64: string, mimeType = "image/png"): string {
  const byteCharacters = atob(base64);
  const byteNumbers = new Array(byteCharacters.length);
  for (let i = 0; i < byteCharacters.length; i++) {
    byteNumbers[i] = byteCharacters.charCodeAt(i);
  }
  const blob = new Blob([new Uint8Array(byteNumbers)], { type: mimeType });
  return URL.createObjectURL(blob);
}
