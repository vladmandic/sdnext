import { useCanvasStore } from "@/stores/canvasStore";

export function useIsImg2Img() {
  const hasLayers = useCanvasStore((s) => s.layers.length > 0);
  const inputRole = useCanvasStore((s) => s.inputRole);
  return hasLayers && inputRole === "initial";
}
