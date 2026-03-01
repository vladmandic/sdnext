import { useCallback } from "react";
import { useCanvasStore } from "@/stores/canvasStore";
import { useShortcut } from "@/hooks/useShortcut";
import type Konva from "konva";

export function useImageTransform(
  stageRef: React.RefObject<Konva.Stage | null>,
  trRef: React.RefObject<Konva.Transformer | null>,
) {
  const onStageClick = useCallback((e: Konva.KonvaEventObject<MouseEvent>) => {
    if (useCanvasStore.getState().activeTool !== "move") return;
    if (e.target === stageRef.current) {
      useCanvasStore.getState().setActiveLayer(null);
      trRef.current?.nodes([]);
    }
  }, [stageRef, trRef]);

  const deleteLayer = useCallback(() => {
    if (useCanvasStore.getState().activeTool !== "move") return;
    const id = useCanvasStore.getState().activeLayerId;
    if (id) {
      useCanvasStore.getState().removeLayer(id);
      trRef.current?.nodes([]);
    }
  }, [trRef]);

  useShortcut("canvas-delete", deleteLayer);
  useShortcut("canvas-delete-backspace", deleteLayer);

  return { onStageClick };
}
