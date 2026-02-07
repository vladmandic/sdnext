import { useRef, useEffect, useState, useCallback } from "react";
import { Stage } from "react-konva";
import { useCanvasStore } from "@/stores/canvasStore";
import { useImg2ImgStore } from "@/stores/img2imgStore";
import { useGenerationStore } from "@/stores/generationStore";
import { usePanZoom } from "./tools/usePanZoom";
import { useMaskPaint } from "./tools/useMaskPaint";
import { ImageLayer } from "./layers/ImageLayer";
import { MaskLayer } from "./layers/MaskLayer";
import { OutputLayer } from "./layers/OutputLayer";
import type Konva from "konva";

const PADDING = 32;
const OUTPUT_GAP = 48;

export function CanvasStage() {
  const containerRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<Konva.Stage>(null);
  const [containerSize, setContainerSize] = useState({ width: 0, height: 0 });
  const viewport = useCanvasStore((s) => s.viewport);
  const setViewport = useCanvasStore((s) => s.setViewport);
  const initImageWidth = useImg2ImgStore((s) => s.initImageWidth);
  const initImageHeight = useImg2ImgStore((s) => s.initImageHeight);
  const genWidth = useGenerationStore((s) => s.width);
  const genHeight = useGenerationStore((s) => s.height);
  const panZoom = usePanZoom(stageRef);
  const maskPaint = useMaskPaint({ stageRef, spaceHeld: panZoom.spaceHeld });

  // Side-by-side derived values
  const outputOffsetX = initImageWidth > 0 ? initImageWidth + OUTPUT_GAP : 0;
  const outputW = genWidth || initImageWidth;
  const outputH = genHeight || initImageHeight;

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

  // Auto-fit both images (side-by-side) to viewport
  const fitToView = useCallback(() => {
    if (initImageWidth <= 0 || initImageHeight <= 0) return;
    if (containerSize.width <= 0 || containerSize.height <= 0) return;

    const totalWidth = initImageWidth + OUTPUT_GAP + outputW;
    const totalHeight = Math.max(initImageHeight, outputH);
    const availW = containerSize.width - PADDING * 2;
    const availH = containerSize.height - PADDING * 2;
    const scale = Math.min(availW / totalWidth, availH / totalHeight, 1);
    const x = (containerSize.width - totalWidth * scale) / 2;
    const y = (containerSize.height - totalHeight * scale) / 2;
    setViewport({ x, y, scale });
  }, [initImageWidth, initImageHeight, outputW, outputH, containerSize, setViewport]);

  // Fit on image load or container resize
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
        >
          <ImageLayer />
          <MaskLayer currentLine={maskPaint.currentLine} cursorPos={maskPaint.cursorPos} />
          <OutputLayer offsetX={outputOffsetX} placeholderWidth={outputW} placeholderHeight={outputH} />
        </Stage>
      )}
    </div>
  );
}
