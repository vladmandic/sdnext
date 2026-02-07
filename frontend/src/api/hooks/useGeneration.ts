import { useMutation, useQuery } from "@tanstack/react-query";
import { api } from "../client";
import type { Txt2ImgRequest, Img2ImgRequest, GenerationResponse } from "../types/generation";
import type { ResProgress, ResStatus } from "../types/progress";

export function useTxt2Img() {
  return useMutation({
    mutationFn: (params: Txt2ImgRequest) =>
      api.post<GenerationResponse>("/sdapi/v1/txt2img", params),
  });
}

export function useImg2Img() {
  return useMutation({
    mutationFn: (params: Img2ImgRequest) =>
      api.post<GenerationResponse>("/sdapi/v1/img2img", params),
  });
}

/**
 * Base URL that bypasses the Vite dev proxy.
 *
 * During generation the txt2img POST blocks the Vite proxy's connection pool,
 * so lightweight polling GETs must go directly to the backend.
 */
function getDirectUrl(path: string): string {
  if (window.location.port === "5173") {
    return `${window.location.protocol}//${window.location.hostname}:7860${path}`;
  }
  return path;
}

/**
 * Polls the progress endpoint directly against the backend.
 *
 * The backend returns progress=0 when job_count==0, even while sampling_step
 * is actively incrementing. We compute progress from the state dict as a
 * fallback so the UI always reflects real sampling progress.
 */
export function useProgress(enabled: boolean) {
  return useQuery({
    queryKey: ["progress"],
    queryFn: async () => {
      const res = await fetch(getDirectUrl("/sdapi/v1/progress"));
      const data = (await res.json()) as ResProgress;
      // Backend may return progress=0 when job_count==0 despite active sampling.
      // Calculate from state dict when the reported progress is 0 but steps are non-zero.
      if (data.progress === 0 && data.state) {
        const step = (data.state.sampling_step as number) ?? 0;
        const steps = (data.state.sampling_steps as number) ?? 0;
        if (step > 0 && steps > 0) {
          data.progress = Math.min(step / steps, 1);
        }
      }
      return data;
    },
    refetchInterval: enabled ? 250 : false,
    staleTime: 0,
    gcTime: 0,
    enabled,
  });
}

export function useStatus() {
  return useQuery({
    queryKey: ["status"],
    queryFn: async () => {
      const res = await fetch(getDirectUrl("/sdapi/v1/status"));
      return (await res.json()) as ResStatus;
    },
    refetchInterval: 2000,
  });
}

export function useInterrupt() {
  return useMutation({
    mutationFn: () => api.post("/sdapi/v1/interrupt"),
  });
}

export function useSkip() {
  return useMutation({
    mutationFn: () => api.post("/sdapi/v1/skip"),
  });
}
