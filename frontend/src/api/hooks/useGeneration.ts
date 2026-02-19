import { useQuery } from "@tanstack/react-query";
import type { ResProgress, ResStatus } from "../types/progress";

/**
 * Base URL that bypasses the Vite dev proxy.
 *
 * During development the Vite proxy on :5173 may not route lightweight polling
 * GETs correctly when other connections are active, so we hit the backend directly.
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
