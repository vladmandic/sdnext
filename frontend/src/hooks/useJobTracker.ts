import { useEffect, useRef } from "react";
import { api } from "@/api/client";
import { WebSocketManager } from "@/api/websocket";
import { useJobQueueStore, type TrackedJob, type JobDomain } from "@/stores/jobStore";
import { useGenerationStore } from "@/stores/generationStore";
import { useVideoStore } from "@/stores/videoStore";
import { useProcessStore } from "@/stores/processStore";
import type { JobResult, JobWsEvent } from "@/api/types/v2";

const MAX_CONCURRENT_WS = 5;

function isTerminal(status: string) {
  return status === "completed" || status === "failed" || status === "cancelled";
}

function routeResult(domain: JobDomain, result: JobResult, snapshot: TrackedJob["snapshot"]) {
  if (domain === "generate") {
    if (result.images.length > 0) {
      useGenerationStore.getState().addResult({
        id: crypto.randomUUID(),
        images: result.images.map((img) => img.url),
        parameters: result.params,
        info: JSON.stringify(result.info),
        timestamp: Date.now(),
        inputImage: snapshot.inputImage,
        inputMask: snapshot.inputMask,
        controlUnits: snapshot.controlUnits,
      });
    }
  } else if (domain === "video" || domain === "framepack" || domain === "ltx") {
    const vid = result.images[0];
    if (vid) {
      useVideoStore.getState().setResultVideo(`${api.getBaseUrl()}${vid.url}`);
    }
  } else if (domain === "upscale") {
    const img = result.images[0];
    if (img) {
      useProcessStore.getState().setResult(`${api.getBaseUrl()}${img.url}`, img.width, img.height);
    }
  }
}

interface WsEntry {
  manager: WebSocketManager;
  offMessage: () => void;
  offBinary: () => void;
}

let wsMapInstance: Map<string, WsEntry> | null = null;

export function sendToJob(jobId: string, data: Record<string, unknown>) {
  wsMapInstance?.get(jobId)?.manager.send(data);
}

export function useJobTracker() {
  const wsMap = useRef(new Map<string, WsEntry>());

  useEffect(() => {
    wsMapInstance = wsMap.current;
    return () => { wsMapInstance = null; };
  }, []);

  useEffect(() => {
    const currentWsMap = wsMap.current;
    const unsub = useJobQueueStore.subscribe((state, prev) => {
      if (state.jobs === prev.jobs) return;

      const store = useJobQueueStore.getState();
      const currentMap = currentWsMap;

      // Close WS for jobs that are gone or terminal
      for (const [id, entry] of currentMap) {
        const job = store.jobs.get(id);
        if (!job || isTerminal(job.status)) {
          entry.offMessage();
          entry.offBinary();
          entry.manager.disconnect();
          currentMap.delete(id);
        }
      }

      // Open WS for non-terminal jobs that don't have one yet
      const nonTerminal = Array.from(store.jobs.values())
        .filter((j) => !isTerminal(j.status) && !currentMap.has(j.id))
        .sort((a, b) => b.createdAt - a.createdAt);

      const slotsAvailable = MAX_CONCURRENT_WS - currentMap.size;
      const toOpen = nonTerminal.slice(0, Math.max(0, slotsAvailable));

      for (const job of toOpen) {
        const wsUrl = api.getWebSocketUrl(`/sdapi/v2/jobs/${job.id}/ws`);
        const manager = new WebSocketManager(wsUrl);
        const jobId = job.id;

        const offMessage = manager.on("message", (raw: unknown) => {
          const data = raw as JobWsEvent;
          const s = useJobQueueStore.getState();
          switch (data.type) {
            case "progress":
              s.updateProgress(jobId, data.progress, data.eta ?? 0, data.step, data.steps);
              break;
            case "status":
              s.updateStatus(jobId, data.status);
              break;
            case "completed":
              s.completeJob(jobId, data.result);
              routeResult(s.jobs.get(jobId)?.domain ?? "generate", data.result, s.jobs.get(jobId)?.snapshot ?? {});
              break;
            case "error":
              s.failJob(jobId, data.error);
              break;
            case "cancelled":
              s.updateStatus(jobId, "cancelled");
              break;
          }
        });

        const offBinary = manager.on("binary", (buf: ArrayBuffer) => {
          const blob = new Blob([buf], { type: "image/jpeg" });
          const url = URL.createObjectURL(blob);
          useJobQueueStore.getState().updatePreview(jobId, url);
        });

        currentMap.set(jobId, { manager, offMessage, offBinary });
        manager.connect();
      }
    });

    return () => {
      unsub();
      for (const [, entry] of currentWsMap) {
        entry.offMessage();
        entry.offBinary();
        entry.manager.disconnect();
      }
      currentWsMap.clear();
    };
  }, []);
}
