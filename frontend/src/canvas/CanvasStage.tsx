import { useRef, useEffect, useState, useCallback } from "react";
import { Stage } from "react-konva";
import { useCanvasStore } from "@/stores/canvasStore";
import { useGenerationStore } from "@/stores/generationStore";
import { usePanZoom } from "./tools/usePanZoom";
import { useMaskPaint } from "./tools/useMaskPaint";
import { useImageTransform } from "./tools/useImageTransform";
import { FrameLayer } from "./layers/FrameLayer";
import { CompositeLayer } from "./layers/CompositeLayer";
import { MaskLayer } from "./layers/MaskLayer";
import { OutputLayer } from "./layers/OutputLayer";
import type Konva from "konva";

const PADDING = 32;
const OUTPUT_GAP = 48;
const LABEL_HEIGHT = 19; // space reserved above images for labels

export function CanvasStage() {
  const containerRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<Konva.Stage>(null);
  const trRef = useRef<Konva.Transformer>(null);
  const [containerSize, setContainerSize] = useState({ width: 0, height: 0 });
  const viewport = useCanvasStore((s) => s.viewport);
  const setViewport = useCanvasStore((s) => s.setViewport);
  const frameW = useGenerationStore((s) => s.width);
  const frameH = useGenerationStore((s) => s.height);
  const panZoom = usePanZoom(stageRef);
  const maskPaint = useMaskPaint({ stageRef, spaceHeld: panZoom.spaceHeld });
  const imageTransform = useImageTransform(stageRef, trRef);

  // Side-by-side derived values
  const outputOffsetX = frameW + OUTPUT_GAP;
  const outputW = frameW;
  const outputH = frameH;

  // Container-responsive sizing
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (entry) {
        setContainerSize({
          width: entry.contentRect.width,
          height: entry.contentRect.height,
        });
      }
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Auto-fit frame + output pair to viewport
  const fitToView = useCallback(() => {
    if (frameW <= 0 || frameH <= 0) return;
    if (containerSize.width <= 0 || containerSize.height <= 0) return;

    const totalWidth = frameW + OUTPUT_GAP + outputW;
    const totalHeight = LABEL_HEIGHT + Math.max(frameH, outputH);
    const availW = containerSize.width - PADDING * 2;
    const availH = containerSize.height - PADDING * 2;
    const scale = Math.min(availW / totalWidth, availH / totalHeight, 1);
    const x = (containerSize.width - totalWidth * scale) / 2;
    const y = (containerSize.height - totalHeight * scale) / 2 + LABEL_HEIGHT * scale;
    setViewport({ x, y, scale });
  }, [frameW, frameH, outputW, outputH, containerSize, setViewport]);

  // Fit on frame change or container resize
  useEffect(() => {
    fitToView();
  }, [fitToView]);

  // Compose event handlers: maskPaint first, then panZoom
  const onMouseDown = useCallback((e: Konva.KonvaEventObject<MouseEvent>) => {
    maskPaint.onMouseDown(e);
    panZoom.onMouseDown(e);
  }, [maskPaint.onMouseDown, panZoom.onMouseDown]);

  const onMouseMove = useCallback((e: Konva.KonvaEventObject<MouseEvent>) => {
    maskPaint.onMouseMove(e);
    panZoom.onMouseMove(e);
  }, [maskPaint.onMouseMove, panZoom.onMouseMove]);

  const onMouseUp = useCallback(() => {
    maskPaint.onMouseUp();
    panZoom.onMouseUp();
  }, [maskPaint.onMouseUp, panZoom.onMouseUp]);

  const onClick = useCallback((e: Konva.KonvaEventObject<MouseEvent>) => {
    imageTransform.onStageClick(e);
  }, [imageTransform.onStageClick]);

  return (
    <div ref={containerRef} className="w-full h-full overflow-hidden">
      {containerSize.width > 0 && containerSize.height > 0 && (
        <Stage
          ref={stageRef}
          width={containerSize.width}
          height={containerSize.height}
          x={viewport.x}
          y={viewport.y}
          scaleX={viewport.scale}
          scaleY={viewport.scale}
          onWheel={panZoom.onWheel}
          onMouseDown={onMouseDown}
          onMouseMove={onMouseMove}
          onMouseUp={onMouseUp}
          onMouseLeave={maskPaint.onMouseLeave}
          onClick={onClick}
        >
          <CompositeLayer trRef={trRef} />
          <FrameLayer />
          <MaskLayer activeLineRef={maskPaint.activeLineRef} cursorRef={maskPaint.cursorRef} />
          <OutputLayer offsetX={outputOffsetX} placeholderWidth={outputW} placeholderHeight={outputH} />
        </Stage>
      )}
    </div>
  );
}
