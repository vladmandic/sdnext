import { useCanvasStore } from "@/stores/canvasStore";

export function useIsImg2Img() {
  return useCanvasStore((s) => s.layers.length > 0);
}
