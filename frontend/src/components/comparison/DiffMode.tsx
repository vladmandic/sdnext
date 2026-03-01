import { useEffect, useRef, useState } from "react";
import { useImageZoomPan } from "@/hooks/useImageZoomPan";
import type { ComparisonImage } from "@/stores/comparisonStore";

interface DiffModeProps {
  imageA: ComparisonImage;
  imageB: ComparisonImage;
}

function computeHeatmap(imgA: HTMLImageElement, imgB: HTMLImageElement): string {
  const w = Math.min(imgA.naturalWidth, imgB.naturalWidth);
  const h = Math.min(imgA.naturalHeight, imgB.naturalHeight);

  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d")!;

  // Draw A
  ctx.drawImage(imgA, 0, 0, w, h);
  const dataA = ctx.getImageData(0, 0, w, h);

  // Draw B
  ctx.clearRect(0, 0, w, h);
  ctx.drawImage(imgB, 0, 0, w, h);
  const dataB = ctx.getImageData(0, 0, w, h);

  const output = ctx.createImageData(w, h);
  const a = dataA.data;
  const b = dataB.data;
  const o = output.data;

  for (let i = 0; i < a.length; i += 4) {
    const diff = (Math.abs(a[i] - b[i]) + Math.abs(a[i + 1] - b[i + 1]) + Math.abs(a[i + 2] - b[i + 2])) / 3;
    const t = diff / 255;

    // Heatmap: black → blue → yellow → red
    let r = 0, g = 0, bl = 0;
    if (t < 0.33) {
      const s = t / 0.33;
      bl = s * 255;
    } else if (t < 0.66) {
      const s = (t - 0.33) / 0.33;
      r = s * 255;
      g = s * 255;
      bl = (1 - s) * 255;
    } else {
      const s = (t - 0.66) / 0.34;
      r = 255;
      g = (1 - s) * 255;
    }

    o[i] = r;
    o[i + 1] = g;
    o[i + 2] = bl;
    o[i + 3] = 255;
  }

  ctx.putImageData(output, 0, 0);
  return canvas.toDataURL("image/png");
}

export function DiffMode({ imageA, imageB }: DiffModeProps) {
  const zoom = useImageZoomPan();
  const [heatmapSrc, setHeatmapSrc] = useState<string | null>(null);
  const computedRef = useRef<string>("");

  useEffect(() => {
    const key = `${imageA.src}|${imageB.src}`;
    if (computedRef.current === key) return;

    const imgA = new Image();
    const imgB = new Image();
    imgA.crossOrigin = "anonymous";
    imgB.crossOrigin = "anonymous";

    let cancelled = false;
    let loaded = 0;
    const onLoad = () => {
      loaded++;
      if (loaded === 2 && !cancelled) {
        const result = computeHeatmap(imgA, imgB);
        computedRef.current = key;
        setHeatmapSrc(result);
      }
    };

    imgA.onload = onLoad;
    imgB.onload = onLoad;
    imgA.src = imageA.src;
    imgB.src = imageB.src;

    return () => { cancelled = true; };
  }, [imageA.src, imageB.src]);

  return (
    <div
      className="relative h-full w-full overflow-hidden flex items-center justify-center select-none"
      onWheel={zoom.handlers.onWheel}
      onMouseDown={zoom.handlers.onMouseDown}
      onMouseMove={zoom.handlers.onMouseMove}
      onMouseUp={zoom.handlers.onMouseUp}
      onMouseLeave={zoom.handlers.onMouseLeave}
      style={{ cursor: zoom.style.cursor }}
    >
      {heatmapSrc ? (
        <img
          src={heatmapSrc}
          alt="Pixel difference"
          className="max-w-full max-h-full object-contain"
          style={{ transform: zoom.style.transform }}
          draggable={false}
        />
      ) : (
        <span className="text-white/50 text-sm">Computing difference...</span>
      )}

      <span className="absolute top-2 left-2 bg-black/60 text-white text-2xs px-2 py-0.5 rounded z-10">
        Pixel Diff
      </span>

      {/* Legend */}
      <div className="absolute bottom-2 left-1/2 -translate-x-1/2 flex items-center gap-2 bg-black/60 px-3 py-1 rounded z-10">
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded-sm bg-black border border-white/20" />
          <span className="text-3xs text-white/70">Same</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded-sm bg-blue-500" />
          <span className="text-3xs text-white/70">Low</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded-sm bg-yellow-400" />
          <span className="text-3xs text-white/70">Medium</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded-sm bg-red-500" />
          <span className="text-3xs text-white/70">High</span>
        </div>
      </div>
    </div>
  );
}
