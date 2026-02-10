import { useRef, useEffect, useState, useCallback, useMemo } from "react";
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
import { ControlFrameLayer } from "./layers/ControlFrameLayer";
import type { CanvasLayout } from "./useControlFrameLayout";
import type Konva from "konva";

const PADDING = 32;
const LABEL_HEIGHT = 19;

interface CanvasStageProps {
  layout: CanvasLayout;
}

export function CanvasStage({ layout }: CanvasStageProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<Konva.Stage>(null);
  const trRef = useRef<Konva.Transformer>(null);
  const [containerSize, setContainerSize] = useState({ width: 0, height: 0 });
  const viewport = useCanvasStore((s) => s.viewport);
  const setViewport = useCanvasStore((s) => s.setViewport);
  const setSelectedControlFrame = useCanvasStore((s) => s.setSelectedControlFrame);
  const frameW = useGenerationStore((s) => s.width);
  const frameH = useGenerationStore((s) => s.height);
  const panZoom = usePanZoom(stageRef);
  const maskPaint = useMaskPaint({ stageRef, spaceHeld: panZoom.spaceHeld });
  const imageTransform = useImageTransform(stageRef, trRef);

  const { showInputFrame, outputX, controlFrames, totalBounds } = layout;

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

  // Fit viewport to show all frames — reads containerSize from ref to avoid
  // re-triggering on minor container resizes (e.g. scrollbar from floating panel).
  const containerSizeRef = useRef(containerSize);
  useEffect(() => { containerSizeRef.current = containerSize; }, [containerSize]);

  const fitToView = useCallback(() => {
    const cs = containerSizeRef.current;
    if (frameW <= 0 || frameH <= 0) return;
    if (cs.width <= 0 || cs.height <= 0) return;

    const totalWidth = totalBounds.maxX - totalBounds.minX;
    const totalHeight = LABEL_HEIGHT + totalBounds.maxY;
    const availW = cs.width - PADDING * 2;
    const availH = cs.height - PADDING * 2;
    const scale = Math.min(availW / totalWidth, availH / totalHeight, 1);
    const x = (cs.width - totalWidth * scale) / 2 - totalBounds.minX * scale;
    const y = (cs.height - totalHeight * scale) / 2 + LABEL_HEIGHT * scale;
    setViewport({ x, y, scale });
  }, [frameW, frameH, totalBounds, setViewport]);

  // Stable key for the logical layout — only refit when this changes.
  const layoutKey = useMemo(
    () => `${frameW},${frameH},${totalBounds.minX},${totalBounds.maxX},${totalBounds.maxY}`,
    [frameW, frameH, totalBounds],
  );

  // Auto-fit on genuine layout changes (frame dimensions, control frame count).
  // Also fit when container first gets a nonzero size (initial render).
  const prevLayoutKeyRef = useRef("");
  const hadSizeRef = useRef(false);
  useEffect(() => {
    const hasSize = containerSize.width > 0 && containerSize.height > 0;
    const layoutChanged = layoutKey !== prevLayoutKeyRef.current;
    const firstSize = hasSize && !hadSizeRef.current;
    if (hasSize) hadSizeRef.current = true;
    if (layoutChanged || firstSize) {
      prevLayoutKeyRef.current = layoutKey;
      fitToView();
    }
  }, [layoutKey, containerSize, fitToView]);

  // Compose event handlers: maskPaint first, then panZoom
  const onMouseDown = useCallback((e: Konva.KonvaEventObject<MouseEvent>) => {
    if (showInputFrame) maskPaint.onMouseDown(e);
    panZoom.onMouseDown(e);
  }, [showInputFrame, maskPaint, panZoom]);

  const onMouseMove = useCallback((e: Konva.KonvaEventObject<MouseEvent>) => {
    if (showInputFrame) maskPaint.onMouseMove(e);
    panZoom.onMouseMove(e);
  }, [showInputFrame, maskPaint, panZoom]);

  const onMouseUp = useCallback(() => {
    if (showInputFrame) maskPaint.onMouseUp();
    panZoom.onMouseUp();
  }, [showInputFrame, maskPaint, panZoom]);

  const onClick = useCallback((e: Konva.KonvaEventObject<MouseEvent>) => {
    if (showInputFrame) imageTransform.onStageClick(e);
    // Clear selected control frame if click was not on a control frame rect
    const target = e.target;
    if (target.name() !== "controlFrame") {
      setSelectedControlFrame(null);
    }
  }, [showInputFrame, imageTransform, setSelectedControlFrame]);

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
          onMouseLeave={showInputFrame ? maskPaint.onMouseLeave : undefined}
          onClick={onClick}
        >
          <ControlFrameLayer frames={controlFrames} />
          {showInputFrame && <CompositeLayer trRef={trRef} />}
          {showInputFrame && <FrameLayer />}
          {showInputFrame && <MaskLayer setActiveLineNode={maskPaint.setActiveLineNode} setCursorNode={maskPaint.setCursorNode} />}
          <OutputLayer offsetX={outputX} placeholderWidth={frameW} placeholderHeight={frameH} />
        </Stage>
      )}
    </div>
  );
}
