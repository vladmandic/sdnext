import { create } from "zustand";

export interface DownloadProgress {
  id: string;
  status: string;
  progress: number;
  filename: string;
  bytes_downloaded: number;
  bytes_total: number;
}

interface DownloadState {
  items: DownloadProgress[];
  updateFromWs: (data: DownloadProgress[]) => void;
}

export const useDownloadStore = create<DownloadState>()((set) => ({
  items: [],
  updateFromWs: (data) => set({ items: data }),
}));
