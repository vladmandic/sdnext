import { useCallback, useId, useState } from "react";
import { useDragStore, INTERNAL_MIME } from "@/stores/dragStore";
import type { DragPayload } from "@/stores/dragStore";

interface UseDropTargetOptions {
  onDropPayload?: (payload: DragPayload, e: React.DragEvent) => void;
  onFileDrop?: (file: File, e: React.DragEvent) => void;
  acceptTypes?: DragPayload["type"][];
}

export function useDropTarget({ onDropPayload, onFileDrop, acceptTypes }: UseDropTargetOptions) {
  const [isOver, setIsOver] = useState(false);
  const targetId = useId();

  const onDragOver = useCallback((e: React.DragEvent) => {
    // Accept internal drags or native file drags
    if (e.dataTransfer.types.includes(INTERNAL_MIME) || e.dataTransfer.types.includes("Files")) {
      e.preventDefault();
      e.dataTransfer.dropEffect = "copy";
    }
  }, []);

  const onDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsOver(true);
    useDragStore.getState().setActiveDropTarget(targetId);
  }, [targetId]);

  const onDragLeave = useCallback((e: React.DragEvent) => {
    // Only deactivate if leaving the target element itself (not a child)
    if (e.currentTarget.contains(e.relatedTarget as Node)) return;
    setIsOver(false);
    const store = useDragStore.getState();
    if (store.activeDropTargetId === targetId) {
      store.setActiveDropTarget(null);
    }
  }, [targetId]);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsOver(false);
    useDragStore.getState().setActiveDropTarget(null);

    // Check for internal drag payload first
    const raw = e.dataTransfer.getData(INTERNAL_MIME);
    if (raw) {
      try {
        const payload = JSON.parse(raw) as DragPayload;
        if (!acceptTypes || acceptTypes.includes(payload.type)) {
          onDropPayload?.(payload, e);
          useDragStore.getState().endDrag();
          return;
        }
      } catch { /* fall through to file handling */ }
    }

    // Fall through to native file drops
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith("image/")) {
      onFileDrop?.(file, e);
    }
  }, [onDropPayload, onFileDrop, acceptTypes]);

  return { onDragOver, onDragEnter, onDragLeave, onDrop, isOver };
}
