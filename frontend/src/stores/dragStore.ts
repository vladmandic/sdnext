import { create } from "zustand";

export interface DragPayload {
  type: "result-image" | "gallery-image";
  resultId?: string;
  imageIndex?: number;
  src?: string;
  fileId?: string;
  filePath?: string;
}

interface DragStoreState {
  isDragging: boolean;
  payload: DragPayload | null;
  activeDropTargetId: string | null;

  startDrag: (payload: DragPayload) => void;
  endDrag: () => void;
  setActiveDropTarget: (id: string | null) => void;
}

export const INTERNAL_MIME = "application/x-sdnext-image";

export const useDragStore = create<DragStoreState>()((set) => ({
  isDragging: false,
  payload: null,
  activeDropTargetId: null,

  startDrag: (payload) => set({ isDragging: true, payload }),
  endDrag: () => set({ isDragging: false, payload: null, activeDropTargetId: null }),
  setActiveDropTarget: (id) => set({ activeDropTargetId: id }),
}));
