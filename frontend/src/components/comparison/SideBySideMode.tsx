import { useImageZoomPanSync } from "@/hooks/useImageZoomPanSync";
import type { ComparisonImage } from "@/stores/comparisonStore";

interface SideBySideModeProps {
  imageA: ComparisonImage;
  imageB: ComparisonImage;
}

export function SideBySideMode({ imageA, imageB }: SideBySideModeProps) {
  const { left, right } = useImageZoomPanSync();

  return (
    <div className="flex h-full w-full gap-1">
      <div
        className="flex-1 overflow-hidden flex items-center justify-center relative"
        onWheel={left.handlers.onWheel}
        onMouseDown={left.handlers.onMouseDown}
        onMouseMove={left.handlers.onMouseMove}
        onMouseUp={left.handlers.onMouseUp}
        onMouseLeave={left.handlers.onMouseLeave}
        style={{ cursor: left.style.cursor }}
      >
        <img
          src={imageA.src}
          alt={imageA.label}
          className="max-w-full max-h-full object-contain"
          style={{ transform: left.style.transform }}
          draggable={false}
        />
        <span className="absolute top-2 left-2 bg-black/60 text-white text-2xs px-2 py-0.5 rounded">{imageA.label}</span>
      </div>
      <div className="w-px bg-white/20 flex-shrink-0" />
      <div
        className="flex-1 overflow-hidden flex items-center justify-center relative"
        onWheel={right.handlers.onWheel}
        onMouseDown={right.handlers.onMouseDown}
        onMouseMove={right.handlers.onMouseMove}
        onMouseUp={right.handlers.onMouseUp}
        onMouseLeave={right.handlers.onMouseLeave}
        style={{ cursor: right.style.cursor }}
      >
        <img
          src={imageB.src}
          alt={imageB.label}
          className="max-w-full max-h-full object-contain"
          style={{ transform: right.style.transform }}
          draggable={false}
        />
        <span className="absolute top-2 left-2 bg-black/60 text-white text-2xs px-2 py-0.5 rounded">{imageB.label}</span>
      </div>
    </div>
  );
}
