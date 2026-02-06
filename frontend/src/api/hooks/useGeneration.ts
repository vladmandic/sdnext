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

export function useProgress(enabled: boolean) {
  return useQuery({
    queryKey: ["progress"],
    queryFn: () => api.get<ResProgress>("/sdapi/v1/progress"),
    refetchInterval: enabled ? 500 : false,
    enabled,
  });
}

export function useStatus() {
  return useQuery({
    queryKey: ["status"],
    queryFn: () => api.get<ResStatus>("/sdapi/v1/status"),
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
