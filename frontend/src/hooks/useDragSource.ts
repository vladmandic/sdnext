import { useCallback, useEffect, useRef } from "react";
import { useDragStore, INTERNAL_MIME } from "@/stores/dragStore";
import type { DragPayload } from "@/stores/dragStore";

export function useDragSource(payload: DragPayload) {
  const payloadRef = useRef(payload);
  useEffect(() => { payloadRef.current = payload; });

  const onDragStart = useCallback((e: React.DragEvent) => {
    const p = payloadRef.current;
    e.dataTransfer.setData(INTERNAL_MIME, JSON.stringify(p));
    e.dataTransfer.effectAllowed = "copy";
    // Set a small drag image from the thumbnail if available
    if (p.src) {
      const img = new Image();
      img.src = p.src;
      e.dataTransfer.setDragImage(img, 24, 24);
    }
    useDragStore.getState().startDrag(p);
  }, []);

  const onDragEnd = useCallback(() => {
    useDragStore.getState().endDrag();
  }, []);

  return { draggable: true as const, onDragStart, onDragEnd };
}
