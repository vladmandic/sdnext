import { useCallback, useEffect, useRef } from "react";
import { useCanvasStore } from "@/stores/canvasStore";
import { ZOOM_LIMITS } from "@/lib/constants";
import type Konva from "konva";

type SetViewportFn = (v: { x?: number; y?: number; scale?: number }) => void;

export function usePanZoom(stageRef: React.RefObject<Konva.Stage | null>, setViewportOverride?: SetViewportFn) {
  const canvasSetViewport = useCanvasStore((s) => s.setViewport);
  const setViewport = setViewportOverride ?? canvasSetViewport;
  const spaceHeld = useRef(false);
  const isPanning = useRef(false);
  const lastPointer = useRef({ x: 0, y: 0 });

  const handleWheel = useCallback((e: Konva.KonvaEventObject<WheelEvent>) => {
    e.evt.preventDefault();
    const stage = stageRef.current;
    if (!stage) return;

    const pointer = stage.getPointerPosition();
    if (!pointer) return;

    const oldScale = stage.scaleX();
    const direction = e.evt.deltaY > 0 ? -1 : 1;
    const factor = 1.1;
    const newScale = Math.min(
      ZOOM_LIMITS.max,
      Math.max(ZOOM_LIMITS.min, direction > 0 ? oldScale * factor : oldScale / factor),
    );

    const mousePointTo = {
      x: (pointer.x - stage.x()) / oldScale,
      y: (pointer.y - stage.y()) / oldScale,
    };

    setViewport({
      scale: newScale,
      x: pointer.x - mousePointTo.x * newScale,
      y: pointer.y - mousePointTo.y * newScale,
    });
  }, [stageRef, setViewport]);

  const handleMouseDown = useCallback((e: Konva.KonvaEventObject<MouseEvent>) => {
    // Middle-click or space+left-click to pan
    if (e.evt.button === 1 || (spaceHeld.current && e.evt.button === 0)) {
      e.evt.preventDefault();
      isPanning.current = true;
      lastPointer.current = { x: e.evt.clientX, y: e.evt.clientY };
      const stage = stageRef.current;
      if (stage) {
        const container = stage.container();
        container.style.cursor = "grabbing";
      }
    }
  }, [stageRef]);

  const handleMouseMove = useCallback((e: Konva.KonvaEventObject<MouseEvent>) => {
    if (!isPanning.current) return;
    const stage = stageRef.current;
    if (!stage) return;

    const dx = e.evt.clientX - lastPointer.current.x;
    const dy = e.evt.clientY - lastPointer.current.y;
    lastPointer.current = { x: e.evt.clientX, y: e.evt.clientY };

    setViewport({
      x: stage.x() + dx,
      y: stage.y() + dy,
    });
  }, [stageRef, setViewport]);

  const handleMouseUp = useCallback(() => {
    if (isPanning.current) {
      isPanning.current = false;
      const stage = stageRef.current;
      if (stage) {
        const container = stage.container();
        container.style.cursor = spaceHeld.current ? "grab" : "default";
      }
    }
  }, [stageRef]);

  // Space key tracking for pan mode
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.code === "Space" && !e.repeat) {
        spaceHeld.current = true;
        const stage = stageRef.current;
        if (stage) stage.container().style.cursor = "grab";
      }
    };
    const onKeyUp = (e: KeyboardEvent) => {
      if (e.code === "Space") {
        spaceHeld.current = false;
        const stage = stageRef.current;
        if (stage && !isPanning.current) stage.container().style.cursor = "default";
      }
    };
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
    };
  }, [stageRef]);

  return {
    onWheel: handleWheel,
    onMouseDown: handleMouseDown,
    onMouseMove: handleMouseMove,
    onMouseUp: handleMouseUp,
    spaceHeld,
  };
}
