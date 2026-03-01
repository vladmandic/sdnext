import { useCallback, useRef } from "react";
import { useCanvasStore } from "@/stores/canvasStore";
import { useImg2ImgStore } from "@/stores/img2imgStore";
import { useGenerationStore } from "@/stores/generationStore";
import { REFERENCE_HEIGHT } from "@/canvas/useControlFrameLayout";
import { bakeMaskStrokes } from "@/lib/bakeMask";
import type { MaskLine } from "@/stores/img2imgStore";
import type Konva from "konva";

interface UseMaskPaintOptions {
  stageRef: React.RefObject<Konva.Stage | null>;
  spaceHeld: React.RefObject<boolean>;
}

/** Compute displayScale imperatively from the store (no reactive dep). */
function getDisplayScale(): number {
  const h = useGenerationStore.getState().height;
  return h > 0 ? REFERENCE_HEIGHT / h : 1;
}

/**
 * Imperative mask painting. During a stroke, points are pushed into a
 * mutable buffer and flushed straight to a Konva Line node — zero React
 * state updates, zero re-renders. React only sees a change on mouseUp
 * when the finished stroke commits to the Zustand store.
 */
export function useMaskPaint({ stageRef, spaceHeld }: UseMaskPaintOptions) {
  const isDrawing = useRef(false);
  const pointsBuffer = useRef<number[]>([]);
  const toolRef = useRef<MaskLine["tool"]>("brush");
  const strokeWidthRef = useRef(20);

  // Konva node refs — MaskLayer attaches these via callback setters
  const activeLineRef = useRef<Konva.Line | null>(null);
  const cursorRef = useRef<Konva.Circle | null>(null);

  const setActiveLineNode = useCallback((node: Konva.Line | null) => {
    activeLineRef.current = node;
  }, []);

  const setCursorNode = useCallback((node: Konva.Circle | null) => {
    cursorRef.current = node;
  }, []);

  const getCanvasPos = useCallback((stage: Konva.Stage): { x: number; y: number } | null => {
    const pointer = stage.getPointerPosition();
    if (!pointer) return null;
    const { width: frameW, height: frameH } = useGenerationStore.getState();
    if (frameW <= 0 || frameH <= 0) return null;
    // screen → stage (display units) → pixel space
    const ds = getDisplayScale();
    return {
      x: Math.max(0, Math.min(frameW, (pointer.x - stage.x()) / stage.scaleX() / ds)),
      y: Math.max(0, Math.min(frameH, (pointer.y - stage.y()) / stage.scaleY() / ds)),
    };
  }, []);

  const isMaskTool = useCallback(() => {
    const tool = useCanvasStore.getState().activeTool;
    return tool === "maskBrush" || tool === "maskEraser";
  }, []);

  const handleMouseDown = useCallback((e: Konva.KonvaEventObject<MouseEvent>) => {
    if (!isMaskTool() || e.evt.button !== 0 || spaceHeld.current) return;
    const stage = stageRef.current;
    if (!stage) return;
    const pos = getCanvasPos(stage);
    if (!pos) return;

    const { activeTool, brushSize } = useCanvasStore.getState();
    toolRef.current = activeTool === "maskEraser" ? "eraser" : "brush";
    strokeWidthRef.current = brushSize;
    pointsBuffer.current = [pos.x, pos.y];
    isDrawing.current = true;

    // Configure the active line node imperatively
    const line = activeLineRef.current;
    if (line) {
      line.points([pos.x, pos.y]);
      line.strokeWidth(brushSize);
      line.globalCompositeOperation(toolRef.current === "eraser" ? "destination-out" : "source-over");
      line.visible(true);
      line.getLayer()?.batchDraw();
    }
  }, [stageRef, spaceHeld, getCanvasPos, isMaskTool]);

  const handleMouseMove = useCallback((_e: Konva.KonvaEventObject<MouseEvent>) => {
    const stage = stageRef.current;
    if (!stage) return;

    if (isMaskTool()) {
      const pos = getCanvasPos(stage);
      const cursor = cursorRef.current;
      if (cursor && pos) {
        // Combined scale: viewport.scale * displayScale (cursor is inside Group)
        const combinedScale = stage.scaleX() * getDisplayScale();
        cursor.x(pos.x);
        cursor.y(pos.y);
        cursor.radius(useCanvasStore.getState().brushSize / 2);
        cursor.strokeWidth(1 / combinedScale);
        cursor.dash([4 / combinedScale, 4 / combinedScale]);
        cursor.visible(true);
      } else if (cursor) {
        cursor.visible(false);
      }
      stage.container().style.cursor = "none";

      if (!isDrawing.current) {
        cursor?.getLayer()?.batchDraw();
      }
    } else {
      // Switched away from mask tool — restore cursor and hide the circle
      const cursor = cursorRef.current;
      if (cursor?.visible()) {
        cursor.visible(false);
        cursor.getLayer()?.batchDraw();
      }
      if (stage.container().style.cursor === "none") {
        stage.container().style.cursor = "";
      }
    }

    if (!isDrawing.current) return;
    const pos = getCanvasPos(stage);
    if (!pos) return;

    pointsBuffer.current.push(pos.x, pos.y);

    const line = activeLineRef.current;
    if (line) {
      line.points(pointsBuffer.current);
      line.getLayer()?.batchDraw();
    }
  }, [stageRef, getCanvasPos, isMaskTool]);

  const commitLine = useCallback(() => {
    if (!isDrawing.current) return;
    isDrawing.current = false;

    if (pointsBuffer.current.length >= 2) {
      useImg2ImgStore.getState().addMaskLine({
        points: pointsBuffer.current.slice(),
        strokeWidth: strokeWidthRef.current,
        tool: toolRef.current,
      });
      // Bake strokes into mask objects (async, fire-and-forget)
      bakeMaskStrokes();
    }

    const line = activeLineRef.current;
    if (line) {
      line.visible(false);
      line.getLayer()?.batchDraw();
    }
  }, []);

  const handleMouseUp = useCallback(() => {
    commitLine();
  }, [commitLine]);

  const handleMouseLeave = useCallback(() => {
    const cursor = cursorRef.current;
    if (cursor) {
      cursor.visible(false);
      cursor.getLayer()?.batchDraw();
    }
    commitLine();
  }, [commitLine]);

  return {
    onMouseDown: handleMouseDown,
    onMouseMove: handleMouseMove,
    onMouseUp: handleMouseUp,
    onMouseLeave: handleMouseLeave,
    setActiveLineNode,
    setCursorNode,
  };
}
