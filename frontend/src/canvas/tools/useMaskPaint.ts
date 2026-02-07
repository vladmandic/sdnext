import { useCallback, useState } from "react";
import { useCanvasStore } from "@/stores/canvasStore";
import { useImg2ImgStore } from "@/stores/img2imgStore";
import type { MaskLine } from "@/stores/img2imgStore";
import type Konva from "konva";

interface UseMaskPaintOptions {
  stageRef: React.RefObject<Konva.Stage | null>;
  spaceHeld: React.RefObject<boolean>;
}

export function useMaskPaint({ stageRef, spaceHeld }: UseMaskPaintOptions) {
  const [currentLine, setCurrentLine] = useState<MaskLine | null>(null);
  const [cursorPos, setCursorPos] = useState<{ x: number; y: number } | null>(null);
  const [isDrawing, setIsDrawing] = useState(false);

  const getCanvasPos = useCallback((stage: Konva.Stage): { x: number; y: number } | null => {
    const pointer = stage.getPointerPosition();
    if (!pointer) return null;
    const { initImageWidth: initW, initImageHeight: initH } = useImg2ImgStore.getState();
    if (initW <= 0 || initH <= 0) return null;
    const x = Math.max(0, Math.min(initW, (pointer.x - stage.x()) / stage.scaleX()));
    const y = Math.max(0, Math.min(initH, (pointer.y - stage.y()) / stage.scaleY()));
    return { x, y };
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

    const activeTool = useCanvasStore.getState().activeTool;
    const brushSize = useCanvasStore.getState().brushSize;
    setIsDrawing(true);
    setCurrentLine({
      points: [pos.x, pos.y],
      strokeWidth: brushSize,
      tool: activeTool === "maskEraser" ? "eraser" : "brush",
    });
  }, [stageRef, spaceHeld, getCanvasPos, isMaskTool]);

  const handleMouseMove = useCallback((_e: Konva.KonvaEventObject<MouseEvent>) => {
    const stage = stageRef.current;
    if (!stage) return;

    // Update cursor position when a mask tool is active
    if (isMaskTool()) {
      const pos = getCanvasPos(stage);
      setCursorPos(pos);

      // Set cursor style
      const container = stage.container();
      if (container) container.style.cursor = "none";
    }

    if (!isDrawing || !currentLine) return;
    const pos = getCanvasPos(stage);
    if (!pos) return;

    setCurrentLine((prev) => {
      if (!prev) return prev;
      return { ...prev, points: [...prev.points, pos.x, pos.y] };
    });
  }, [stageRef, isDrawing, currentLine, getCanvasPos, isMaskTool]);

  const handleMouseUp = useCallback(() => {
    if (!isDrawing || !currentLine) return;
    // Commit the line to the store
    if (currentLine.points.length >= 2) {
      useImg2ImgStore.getState().addMaskLine(currentLine);
    }
    setCurrentLine(null);
    setIsDrawing(false);
  }, [isDrawing, currentLine]);

  const handleMouseLeave = useCallback(() => {
    setCursorPos(null);
    if (isDrawing && currentLine) {
      // Commit partial line on leave
      if (currentLine.points.length >= 2) {
        useImg2ImgStore.getState().addMaskLine(currentLine);
      }
      setCurrentLine(null);
      setIsDrawing(false);
    }
  }, [isDrawing, currentLine]);

  return {
    onMouseDown: handleMouseDown,
    onMouseMove: handleMouseMove,
    onMouseUp: handleMouseUp,
    onMouseLeave: handleMouseLeave,
    currentLine,
    cursorPos,
  };
}
