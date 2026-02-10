import { useMemo } from "react";
import { useControlStore } from "@/stores/controlStore";
import { useGenerationStore } from "@/stores/generationStore";
import { useUiStore } from "@/stores/uiStore";

const FRAME_GAP = 48;

export interface ControlFramePosition {
  unitIndex: number;
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface CanvasLayout {
  showInputFrame: boolean;
  inputX: number;
  outputX: number;
  controlFrames: ControlFramePosition[];
  totalBounds: { minX: number; maxX: number; maxY: number };
}

export function useControlFrameLayout(): CanvasLayout {
  const units = useControlStore((s) => s.units);
  const frameW = useGenerationStore((s) => s.width);
  const frameH = useGenerationStore((s) => s.height);
  const generationMode = useUiStore((s) => s.generationMode);

  return useMemo(() => {
    const isImg2Img = generationMode === "img2img";

    // Main frame positions
    const inputX = 0;
    const outputX = isImg2Img ? frameW + FRAME_GAP : 0;

    // Rightmost edge of main frames
    const mainMaxX = isImg2Img ? outputX + frameW : frameW;

    // Control frames: only enabled units with useSeparateImage
    const activeControlIndices = units
      .map((u, i) => ({ unit: u, index: i }))
      .filter((entry) => entry.unit.enabled && entry.unit.useSeparateImage);

    // Stack control frames right-to-left, starting from the left of the leftmost main frame
    const controlFrames: ControlFramePosition[] = activeControlIndices.map((entry, slot) => ({
      unitIndex: entry.index,
      x: -(slot + 1) * (frameW + FRAME_GAP),
      y: 0,
      width: frameW,
      height: frameH,
    }));

    const minX = controlFrames.length > 0
      ? controlFrames[controlFrames.length - 1].x
      : 0;

    return {
      showInputFrame: isImg2Img,
      inputX,
      outputX,
      controlFrames,
      totalBounds: { minX, maxX: mainMaxX, maxY: frameH },
    };
  }, [units, frameW, frameH, generationMode]);
}
