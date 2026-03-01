import { useMemo } from "react";
import { useVideoStore } from "@/stores/videoStore";
import { REFERENCE_HEIGHT } from "./useControlFrameLayout";

const FRAME_GAP = 48;

export interface VideoCanvasLayout {
  initX: number;
  lastX: number;
  outputX: number;
  totalBounds: { minX: number; maxX: number; maxY: number };
  displayScale: number;
  displayW: number;
  displayH: number;
}

export function useVideoFrameLayout(): VideoCanvasLayout {
  const width = useVideoStore((s) => s.width);
  const height = useVideoStore((s) => s.height);

  return useMemo(() => {
    const ds = height > 0 ? REFERENCE_HEIGHT / height : 1;
    const dw = width * ds;
    const dh = REFERENCE_HEIGHT;

    const initX = 0;
    const lastX = dw + FRAME_GAP;
    const outputX = 2 * (dw + FRAME_GAP);

    return {
      initX,
      lastX,
      outputX,
      totalBounds: { minX: 0, maxX: outputX + dw, maxY: dh },
      displayScale: ds,
      displayW: dw,
      displayH: dh,
    };
  }, [width, height]);
}
