import { useEffect } from "react";
import { ws, ensureWs } from "@/api/wsManager";
import { useBackendStatusStore } from "@/stores/backendStatusStore";
import { useDownloadStore } from "@/stores/downloadStore";

export function useGlobalWs() {
  useEffect(() => {
    ensureWs();

    const offOpen = ws.on("open", () => {
      useBackendStatusStore.getState().setConnected(true);
    });

    const offClose = ws.on("close", () => {
      useBackendStatusStore.getState().setConnected(false);
    });

    const offMessage = ws.on("message", (raw: unknown) => {
      const msg = raw as { type: string; data?: unknown };
      const store = useBackendStatusStore.getState();

      if (msg.type === "progress" && msg.data) {
        store.setStatus(msg.data as Record<string, unknown>);
      } else if (msg.type === "status" && msg.data) {
        store.setStatus(msg.data as Record<string, unknown>);
      } else if (msg.type === "download" && Array.isArray(msg.data)) {
        useDownloadStore.getState().updateFromWs(msg.data);
      }
    });

    const offBinary = ws.on("binary", (buf: ArrayBuffer) => {
      const blob = new Blob([buf], { type: "image/jpeg" });
      const url = URL.createObjectURL(blob);
      useBackendStatusStore.getState().setPreview(url);
    });

    return () => {
      offOpen();
      offClose();
      offMessage();
      offBinary();
    };
  }, []);
}
