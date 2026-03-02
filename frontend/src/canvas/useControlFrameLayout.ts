import { useMemo } from "react";
import { useControlStore } from "@/stores/controlStore";
import { useGenerationStore } from "@/stores/generationStore";
import { useUiStore } from "@/stores/uiStore";
import { useCanvasStore } from "@/stores/canvasStore";
import { useImg2ImgStore } from "@/stores/img2imgStore";
import { resolveGenerationSize } from "@/lib/sizeCompute";

/** Reference height for display-unit normalization: all frames are laid out as
 *  if the main frame were this many units tall. Keeps UI panels consistently
 *  sized regardless of the actual generation resolution. */
export const REFERENCE_HEIGHT = 512;

const FRAME_GAP = 48;
/** Spacing between a frame and its associated elements (processed image, floating panel) */
export const ELEMENT_GAP = 16;
/** Height of the per-unit processed image header bar (matches HEADER_HEIGHT in ControlFramePanel) */
export const PROCESSED_HEADER_HEIGHT = 36;

export interface ProcessedSlot {
  unitIndex: number;
  hasProcessed: boolean;
}

export interface ControlFramePosition {
  unitIndex: number;
  unifiedIndex: number;
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
  /** Factor to convert pixel coords → display coords: displayCoord = pixelCoord * displayScale */
  displayScale: number;
  /** Main frame dimensions in display units */
  displayW: number;
  displayH: number;
}

export function useControlFrameLayout(): CanvasLayout {
  const units = useControlStore((s) => s.units);
  const compositeProcessed = useControlStore((s) => s.compositeProcessed);
  const frameW = useGenerationStore((s) => s.width);
  const frameH = useGenerationStore((s) => s.height);
  const hasLayers = useCanvasStore((s) => s.layers.length > 0);
  const autoFitFrame = useUiStore((s) => s.autoFitFrame);
  const sizeMode = useImg2ImgStore((s) => s.sizeMode);
  const scaleFactor = useImg2ImgStore((s) => s.scaleFactor);
  const megapixelTarget = useImg2ImgStore((s) => s.megapixelTarget);

  return useMemo(() => {
    const isAutoFit = hasLayers && autoFitFrame;
    const effectiveSizeMode = isAutoFit ? sizeMode : "fixed";
    const genSize = resolveGenerationSize(effectiveSizeMode, frameW, frameH, scaleFactor, megapixelTarget);

    // Normalize all layout positions to display units so that UI panels
    // stay a consistent size regardless of the actual generation resolution.
    const ds = frameH > 0 ? REFERENCE_HEIGHT / frameH : 1;
    const dw = frameW * ds;
    const dh = REFERENCE_HEIGHT;

    // Input frame always visible — always at x=0 (display units)
    const inputX = 0;
    const outputX = dw + FRAME_GAP;

    // Processed composite frame: visible when backend composite or any per-unit processedImage exists
    const hasAnyProcessed = !!compositeProcessed || units.some((u) => u.enabled && !!u.processedImage);
    const processedX = outputX + dw + FRAME_GAP;

    // Rightmost edge of main frames (display units)
    const mainMaxX = hasAnyProcessed
      ? processedX + dw
      : outputX + dw;

    // Control frames: only enabled units with imageSource === "separate"
    const activeControlIndices = units
      .map((u, i) => ({ unit: u, index: i }))
      .filter((entry) => entry.unit.enabled && entry.unit.imageSource === "separate");

    // Build control frames with processedSlots — accumulate X to handle variable widths
    const controlFrames: ControlFramePosition[] = [];
    let cursorX = 0;
    for (const entry of activeControlIndices) {
      // Collect all units that share this frame: the owner + any "unit:N" referencing it
      const slots: ProcessedSlot[] = [
        { unitIndex: entry.index, hasProcessed: !!entry.unit.processedImage },
      ];
      for (let i = 0; i < units.length; i++) {
        if (i !== entry.index && units[i].enabled && units[i].imageSource === `unit:${entry.index}`) {
          slots.push({ unitIndex: i, hasProcessed: !!units[i].processedImage });
        }
      }

      // Control frames always match the main frame size — the backend resizes
      // control images to the generation resolution before processing.
      const size = { width: dw, height: dh };

      cursorX -= size.width + FRAME_GAP;
      controlFrames.push({
        unitIndex: entry.index,
        unifiedIndex: entry.index + 2,
        x: cursorX,
        y: 0,
        width: size.width,
        height: size.height,
        processedSlots: slots,
      });
    }

    const minX = controlFrames.length > 0
      ? controlFrames[controlFrames.length - 1].x
      : 0;

    // maxY: account for per-frame height + stacked processed slots (display units)
    let maxY = dh;
    for (const f of controlFrames) {
      const activeSlots = f.processedSlots.filter((s) => s.hasProcessed).length;
      const frameMaxY = f.height + activeSlots * (ELEMENT_GAP + PROCESSED_HEADER_HEIGHT + f.height);
      if (frameMaxY > maxY) maxY = frameMaxY;
    }

    return {
      showInputFrame: true,
      inputX,
      outputX,
      processedX,
      showProcessedFrame: hasAnyProcessed,
      controlFrames,
      totalBounds: { minX, maxX: mainMaxX, maxY },
      genSize,
      displayScale: ds,
      displayW: dw,
      displayH: dh,
    };
  }, [units, compositeProcessed, frameW, frameH, hasLayers, autoFitFrame, sizeMode, scaleFactor, megapixelTarget]);
}
