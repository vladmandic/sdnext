import { useEffect, useCallback } from "react";
import { useCanvasStore } from "@/stores/canvasStore";
import type Konva from "konva";

export function useImageTransform(
  stageRef: React.RefObject<Konva.Stage | null>,
  trRef: React.RefObject<Konva.Transformer | null>,
) {
  const onStageClick = useCallback((e: Konva.KonvaEventObject<MouseEvent>) => {
    if (useCanvasStore.getState().activeTool !== "move") return;
    // Only deselect when clicking the stage background itself
    if (e.target === stageRef.current) {
      useCanvasStore.getState().setActiveLayer(null);
      trRef.current?.nodes([]);
    }
  }, [stageRef, trRef]);

  // Delete/Backspace key removes selected layer
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      if (useCanvasStore.getState().activeTool !== "move") return;

      if (e.key === "Delete" || e.key === "Backspace") {
        const id = useCanvasStore.getState().activeLayerId;
        if (id) {
          useCanvasStore.getState().removeLayer(id);
          trRef.current?.nodes([]);
        }
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [trRef]);

  return { onStageClick };
}
