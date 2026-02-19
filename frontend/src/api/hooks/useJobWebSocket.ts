import { useEffect, useRef, useState } from "react";
import { api } from "../client";
import { WebSocketManager } from "../websocket";
import type { JobResult, JobStatus, JobWsEvent } from "../types/v2";

interface JobProgress {
  step: number;
  steps: number;
  progress: number;
  eta: number | null;
}

export function useJobWebSocket(jobId: string | null) {
  const [progress, setProgress] = useState<JobProgress>({ step: 0, steps: 0, progress: 0, eta: null });
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<JobResult | null>(null);
  const [status, setStatus] = useState<JobStatus>("pending");
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocketManager | null>(null);
  const previewRef = useRef<string | null>(null);

  useEffect(() => {
    if (!jobId) return;

    const wsUrl = api.getWebSocketUrl(`/sdapi/v2/jobs/${jobId}/ws`);
    const wsManager = new WebSocketManager(wsUrl);
    wsRef.current = wsManager;

    const offMessage = wsManager.on("message", (raw: unknown) => {
      const data = raw as JobWsEvent;
      switch (data.type) {
        case "progress":
          setProgress({ step: data.step, steps: data.steps, progress: data.progress, eta: data.eta });
          break;
        case "status":
          setStatus(data.status);
          break;
        case "completed":
          setResult(data.result);
          setStatus("completed");
          break;
        case "error":
          setError(data.error);
          setStatus("failed");
          break;
        case "cancelled":
          setStatus("cancelled");
          break;
      }
    });

    const offBinary = wsManager.on("binary", (buf: ArrayBuffer) => {
      const blob = new Blob([buf], { type: "image/jpeg" });
      const url = URL.createObjectURL(blob);
      if (previewRef.current) URL.revokeObjectURL(previewRef.current);
      previewRef.current = url;
      setPreview(url);
    });

    wsManager.connect();

    return () => {
      offMessage();
      offBinary();
      wsManager.disconnect();
      if (previewRef.current) {
        URL.revokeObjectURL(previewRef.current);
        previewRef.current = null;
      }
    };
  }, [jobId]);

  const send = (data: string | Record<string, unknown>) => wsRef.current?.send(data);

  return { progress, preview, result, status, error, send };
}
