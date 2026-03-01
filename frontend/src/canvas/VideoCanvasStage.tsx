import { useRef, useEffect, useState, useCallback } from "react";
import { Stage } from "react-konva";
import { useVideoCanvasStore } from "@/stores/videoCanvasStore";
import { useVideoStore } from "@/stores/videoStore";
import { usePanZoom } from "./tools/usePanZoom";
import { VideoFrameLayer } from "./layers/VideoFrameLayer";
import { VideoOutputFrame } from "./layers/VideoOutputFrame";
import type { VideoCanvasLayout } from "./useVideoFrameLayout";
import type Konva from "konva";

const PADDING = 32;
const LABEL_HEIGHT = 19;

interface VideoCanvasStageProps {
  layout: VideoCanvasLayout;
  onPickImage?: (which: "init" | "last") => void;
}

export function VideoCanvasStage({ layout, onPickImage }: VideoCanvasStageProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<Konva.Stage>(null);
  const [containerSize, setContainerSize] = useState({ width: 0, height: 0 });
  const viewport = useVideoCanvasStore((s) => s.viewport);
  const setViewport = useVideoCanvasStore((s) => s.setViewport);
  const frameW = useVideoStore((s) => s.width);
  const frameH = useVideoStore((s) => s.height);

  const panZoom = usePanZoom(stageRef, setViewport);

  const { initX, lastX, outputX, totalBounds, displayW, displayH } = layout;

  // Container-responsive sizing
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (entry) {
        setContainerSize({ width: entry.contentRect.width, height: entry.contentRect.height });
      }
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Fit viewport to show all three frames on initial render / size change
  const prevFrameRef = useRef<string>("");
  useEffect(() => {
    if (frameW <= 0 || frameH <= 0) return;
    if (containerSize.width <= 0 || containerSize.height <= 0) return;
    const key = `${frameW}x${frameH}`;
    if (prevFrameRef.current === key) return;
    prevFrameRef.current = key;

    const totalWidth = totalBounds.maxX - totalBounds.minX;
    const totalHeight = LABEL_HEIGHT + totalBounds.maxY;
    const availW = containerSize.width - PADDING * 2;
    const availH = containerSize.height - PADDING * 2;
    const scale = Math.min(availW / totalWidth, availH / totalHeight, 1);
    const x = (containerSize.width - totalWidth * scale) / 2 - totalBounds.minX * scale;
    const y = (containerSize.height - totalHeight * scale) / 2 + LABEL_HEIGHT * scale;
    setViewport({ x, y, scale });
  }, [containerSize, frameW, frameH, totalBounds, setViewport]);

  const onMouseDown = useCallback((e: Konva.KonvaEventObject<MouseEvent>) => {
    panZoom.onMouseDown(e);
  }, [panZoom]);

  const onMouseMove = useCallback((e: Konva.KonvaEventObject<MouseEvent>) => {
    panZoom.onMouseMove(e);
  }, [panZoom]);

  const onMouseUp = useCallback(() => {
    panZoom.onMouseUp();
  }, [panZoom]);

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
        >
          <VideoFrameLayer which="init" offsetX={initX} width={displayW} height={displayH} onPickImage={() => onPickImage?.("init")} />
          <VideoFrameLayer which="last" offsetX={lastX} width={displayW} height={displayH} onPickImage={() => onPickImage?.("last")} />
          <VideoOutputFrame offsetX={outputX} width={displayW} height={displayH} />
        </Stage>
      )}
    </div>
  );
}
