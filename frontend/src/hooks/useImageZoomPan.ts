import { useCallback, useRef, useState } from "react";

interface UseImageZoomPanOptions {
  minScale?: number;
  maxScale?: number;
  wheelFactor?: number;
}

interface UseImageZoomPanReturn {
  scale: number;
  translate: { x: number; y: number };
  isDragging: boolean;
  setScale: (fn: number | ((s: number) => number)) => void;
  resetTransform: () => void;
  handlers: {
    onWheel: (e: React.WheelEvent) => void;
    onMouseDown: (e: React.MouseEvent) => void;
    onMouseMove: (e: React.MouseEvent) => void;
    onMouseUp: () => void;
    onMouseLeave: () => void;
  };
  style: React.CSSProperties;
}

export type { UseImageZoomPanReturn };

export function useImageZoomPan(options: UseImageZoomPanOptions = {}): UseImageZoomPanReturn {
  const { minScale = 0.25, maxScale = 8, wheelFactor = 1.15 } = options;

  const [scale, setScaleRaw] = useState(1);
  const [translate, setTranslate] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const dragStart = useRef({ x: 0, y: 0 });

  const clamp = useCallback((v: number) => Math.max(minScale, Math.min(maxScale, v)), [minScale, maxScale]);

  const setScale = useCallback((fn: number | ((s: number) => number)) => {
    setScaleRaw((prev) => {
      const next = typeof fn === "function" ? fn(prev) : fn;
      return clamp(next);
    });
  }, [clamp]);

  const resetTransform = useCallback(() => {
    setScaleRaw(1);
    setTranslate({ x: 0, y: 0 });
  }, []);

  const onWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? wheelFactor : 1 / wheelFactor;
    setScaleRaw((s) => clamp(s * factor));
  }, [wheelFactor, clamp]);

  const onMouseDown = useCallback((e: React.MouseEvent) => {
    if (scale <= 1) return;
    setIsDragging(true);
    dragStart.current = { x: e.clientX - translate.x, y: e.clientY - translate.y };
  }, [scale, translate]);

  const onMouseMove = useCallback((e: React.MouseEvent) => {
    if (!isDragging) return;
    setTranslate({ x: e.clientX - dragStart.current.x, y: e.clientY - dragStart.current.y });
  }, [isDragging]);

  const onMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  const style: React.CSSProperties = {
    transform: `translate(${translate.x}px, ${translate.y}px) scale(${scale})`,
    cursor: scale > 1 ? (isDragging ? "grabbing" : "grab") : "default",
  };

  return {
    scale,
    translate,
    isDragging,
    setScale,
    resetTransform,
    handlers: { onWheel, onMouseDown, onMouseMove, onMouseUp, onMouseLeave: onMouseUp },
    style,
  };
}
