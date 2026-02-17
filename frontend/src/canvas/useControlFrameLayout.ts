import { useMemo } from "react";
import { useControlStore } from "@/stores/controlStore";
import { useGenerationStore } from "@/stores/generationStore";
import { useUiStore } from "@/stores/uiStore";
import { useImg2ImgStore } from "@/stores/img2imgStore";
import { resolveGenerationSize } from "@/lib/sizeCompute";

const FRAME_GAP = 48;
/** Spacing between a frame and its associated elements (processed image, floating panel) */
export const ELEMENT_GAP = 16;

export interface ProcessedSlot {
  unitIndex: number;
  hasProcessed: boolean;
}

export interface ControlFramePosition {
  unitIndex: number;
  x: number;
  y: number;
  width: number;
  height: number;
  processedSlots: ProcessedSlot[];
}

export interface CanvasLayout {
  showInputFrame: boolean;
  inputX: number;
  outputX: number;
  processedX: number;
  showProcessedFrame: boolean;
  controlFrames: ControlFramePosition[];
  totalBounds: { minX: number; maxX: number; maxY: number };
  /** Generation size (may differ from frame size when scale/megapixel is active) */
  genSize: { width: number; height: number };
}

export function useControlFrameLayout(): CanvasLayout {
  const units = useControlStore((s) => s.units);
  const compositeProcessed = useControlStore((s) => s.compositeProcessed);
  const frameW = useGenerationStore((s) => s.width);
  const frameH = useGenerationStore((s) => s.height);
  const generationMode = useUiStore((s) => s.generationMode);
  const autoFitFrame = useUiStore((s) => s.autoFitFrame);
  const sizeMode = useImg2ImgStore((s) => s.sizeMode);
  const scaleFactor = useImg2ImgStore((s) => s.scaleFactor);
  const megapixelTarget = useImg2ImgStore((s) => s.megapixelTarget);

  return useMemo(() => {
    const isImg2Img = generationMode === "img2img";
    const isAutoFit = isImg2Img && autoFitFrame;
    const effectiveSizeMode = isAutoFit ? sizeMode : "fixed";
    const genSize = resolveGenerationSize(effectiveSizeMode, frameW, frameH, scaleFactor, megapixelTarget);

    // Main frame positions — Output and Processed use generation size
    const inputX = 0;
    const outputX = isImg2Img ? frameW + FRAME_GAP : 0;

    // Processed composite frame: visible when backend composite or any per-unit processedImage exists
    const hasAnyProcessed = !!compositeProcessed || units.some((u) => u.enabled && !!u.processedImage);
    const processedX = (isImg2Img ? outputX + genSize.width : frameW) + FRAME_GAP;

    // Rightmost edge of main frames
    const mainMaxX = hasAnyProcessed
      ? processedX + genSize.width
      : (isImg2Img ? outputX + genSize.width : frameW);

    // Control frames: only enabled units with imageSource === "separate"
    const activeControlIndices = units
      .map((u, i) => ({ unit: u, index: i }))
      .filter((entry) => entry.unit.enabled && entry.unit.imageSource === "separate");

    // Build control frames with processedSlots
    const controlFrames: ControlFramePosition[] = activeControlIndices.map((entry, slot) => {
      // Collect all units that share this frame: the owner + any "unit:N" referencing it
      const slots: ProcessedSlot[] = [
        { unitIndex: entry.index, hasProcessed: !!entry.unit.processedImage },
      ];
      for (let i = 0; i < units.length; i++) {
        if (i !== entry.index && units[i].enabled && units[i].imageSource === `unit:${entry.index}`) {
          slots.push({ unitIndex: i, hasProcessed: !!units[i].processedImage });
        }
      }

      return {
        unitIndex: entry.index,
        x: -(slot + 1) * (frameW + FRAME_GAP),
        y: 0,
        width: frameW,
        height: frameH,
        processedSlots: slots,
      };
    });

    const minX = controlFrames.length > 0
      ? controlFrames[controlFrames.length - 1].x
      : 0;

    // maxY: account for the frame with the most processed slots
    const maxSlots = controlFrames.reduce((max, f) => {
      const active = f.processedSlots.filter((s) => s.hasProcessed).length;
      return Math.max(max, active);
    }, 0);
    const tallestFrame = Math.max(frameH, genSize.height);
    const maxY = maxSlots > 0 ? tallestFrame + maxSlots * (ELEMENT_GAP + frameH) : tallestFrame;

    return {
      showInputFrame: isImg2Img,
      inputX,
      outputX,
      processedX,
      showProcessedFrame: hasAnyProcessed,
      controlFrames,
      totalBounds: { minX, maxX: mainMaxX, maxY },
      genSize,
    };
  }, [units, compositeProcessed, frameW, frameH, generationMode, autoFitFrame, sizeMode, scaleFactor, megapixelTarget]);
}
