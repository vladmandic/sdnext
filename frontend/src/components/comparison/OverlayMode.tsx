import { useCallback, useEffect, useState } from "react";
import { useImageZoomPan } from "@/hooks/useImageZoomPan";
import type { ComparisonImage } from "@/stores/comparisonStore";

interface OverlayModeProps {
  imageA: ComparisonImage;
  imageB: ComparisonImage;
}

export function OverlayMode({ imageA, imageB }: OverlayModeProps) {
  const zoom = useImageZoomPan();
  const [showB, setShowB] = useState(false);

  const toggle = useCallback(() => setShowB((v) => !v), []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === " ") {
        e.preventDefault();
        toggle();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [toggle]);

  const current = showB ? imageB : imageA;

  return (
    <div
      className="relative h-full w-full overflow-hidden flex items-center justify-center select-none"
      onWheel={zoom.handlers.onWheel}
      onMouseDown={zoom.handlers.onMouseDown}
      onMouseMove={zoom.handlers.onMouseMove}
      onMouseUp={zoom.handlers.onMouseUp}
      onMouseLeave={zoom.handlers.onMouseLeave}
      onClick={toggle}
      style={{ cursor: zoom.scale > 1 ? zoom.style.cursor : "pointer" }}
    >
      {/* Image A */}
      <img
        src={imageA.src}
        alt={imageA.label}
        className="absolute max-w-full max-h-full object-contain transition-opacity duration-150"
        style={{ transform: zoom.style.transform, opacity: showB ? 0 : 1 }}
        draggable={false}
      />
      {/* Image B */}
      <img
        src={imageB.src}
        alt={imageB.label}
        className="absolute max-w-full max-h-full object-contain transition-opacity duration-150"
        style={{ transform: zoom.style.transform, opacity: showB ? 1 : 0 }}
        draggable={false}
      />

      {/* Label badge */}
      <span className="absolute top-2 left-2 bg-black/60 text-white text-2xs px-2 py-0.5 rounded z-10">
        {current.label} {showB ? "(B)" : "(A)"}
      </span>
      <span className="absolute bottom-2 left-1/2 -translate-x-1/2 bg-black/60 text-white text-3xs px-2 py-0.5 rounded z-10">
        Click or press Space to toggle
      </span>
    </div>
  );
}
