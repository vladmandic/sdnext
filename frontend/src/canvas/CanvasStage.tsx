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
import { ProcessedCompositeLayer } from "./layers/ProcessedCompositeLayer";
import { ControlFrameLayer } from "./layers/ControlFrameLayer";
import type { CanvasLayout } from "./useControlFrameLayout";
import type Konva from "konva";

const PADDING = 32;
const LABEL_HEIGHT = 19;

interface CanvasStageProps {
  layout: CanvasLayout;
  onPickImage?: (unitIndex: number) => void;
}

export function CanvasStage({ layout, onPickImage }: CanvasStageProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<Konva.Stage>(null);
  const trRef = useRef<Konva.Transformer>(null);
  const [containerSize, setContainerSize] = useState({ width: 0, height: 0 });
  const viewport = useCanvasStore((s) => s.viewport);
  const setViewport = useCanvasStore((s) => s.setViewport);
  // Note: setSelectedControlFrame removed - panels are now persistent
  const frameW = useGenerationStore((s) => s.width);
  const frameH = useGenerationStore((s) => s.height);

  const panZoom = usePanZoom(stageRef);
  const maskPaint = useMaskPaint({ stageRef, spaceHeld: panZoom.spaceHeld });
  const imageTransform = useImageTransform(stageRef, trRef);

  const { outputX, processedX, showProcessedFrame, controlFrames, totalBounds, displayScale, displayW, displayH } = layout;

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

  // Fit viewport to show all frames — only runs once on initial render
  const hasFittedRef = useRef(false);
  useEffect(() => {
    if (hasFittedRef.current) return;
    if (frameW <= 0 || frameH <= 0) return;
    if (containerSize.width <= 0 || containerSize.height <= 0) return;
    hasFittedRef.current = true;

    const totalWidth = totalBounds.maxX - totalBounds.minX;
    const totalHeight = LABEL_HEIGHT + totalBounds.maxY;
    const availW = containerSize.width - PADDING * 2;
    const availH = containerSize.height - PADDING * 2;
    const scale = Math.min(availW / totalWidth, availH / totalHeight, 1);
    const x = (containerSize.width - totalWidth * scale) / 2 - totalBounds.minX * scale;
    const y = (containerSize.height - totalHeight * scale) / 2 + LABEL_HEIGHT * scale;
    setViewport({ x, y, scale });
  }, [containerSize, frameW, frameH, totalBounds, setViewport]);

  // Compose event handlers: maskPaint first, then panZoom
  const onMouseDown = useCallback((e: Konva.KonvaEventObject<MouseEvent>) => {
    maskPaint.onMouseDown(e);
    panZoom.onMouseDown(e);
  }, [maskPaint, panZoom]);

  const onMouseMove = useCallback((e: Konva.KonvaEventObject<MouseEvent>) => {
    maskPaint.onMouseMove(e);
    panZoom.onMouseMove(e);
  }, [maskPaint, panZoom]);

  const onMouseUp = useCallback(() => {
    maskPaint.onMouseUp();
    panZoom.onMouseUp();
  }, [maskPaint, panZoom]);

  const onClick = useCallback((e: Konva.KonvaEventObject<MouseEvent>) => {
    if (e.evt.button !== 0) return;
    imageTransform.onStageClick(e);
  }, [imageTransform]);

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
          <ControlFrameLayer frames={controlFrames} onPickImage={onPickImage} />
          <CompositeLayer trRef={trRef} displayScale={displayScale} />
          <FrameLayer displayScale={displayScale} onPickImage={onPickImage ? () => onPickImage(-1) : undefined} />
          <MaskLayer displayScale={displayScale} setActiveLineNode={maskPaint.setActiveLineNode} setCursorNode={maskPaint.setCursorNode} />
          <OutputLayer offsetX={outputX} placeholderWidth={displayW} placeholderHeight={displayH} />
          {showProcessedFrame && (
            <ProcessedCompositeLayer offsetX={processedX} width={displayW} height={displayH} />
          )}
        </Stage>
      )}
    </div>
  );
}
