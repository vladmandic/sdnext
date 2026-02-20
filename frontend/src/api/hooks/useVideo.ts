import { useQuery, useMutation } from "@tanstack/react-query";
import { api } from "../client";
import type { VideoEngine, VideoLoadResponse } from "../types/video";

export function useVideoEngines() {
  return useQuery({
    queryKey: ["video-engines"],
    queryFn: () => api.get<VideoEngine[]>("/sdapi/v2/video/engines"),
    staleTime: 300_000,
  });
}

export function useLoadVideoModel() {
  return useMutation({
    mutationFn: (params: { engine: string; model: string }) =>
      api.post<VideoLoadResponse>("/sdapi/v2/video/load", params),
  });
}

export function useFramePackVariants() {
  return useQuery({
    queryKey: ["framepack-variants"],
    queryFn: () => api.get<string[]>("/sdapi/v2/framepack/variants"),
    staleTime: 300_000,
  });
}

export function useLoadFramePack() {
  return useMutation({
    mutationFn: (params: { variant: string; attention: string }) =>
      api.post<{ variant: string; messages: string[] }>("/sdapi/v2/framepack/load", params),
  });
}

export function useUnloadFramePack() {
  return useMutation({
    mutationFn: () =>
      api.post<{ messages: string[] }>("/sdapi/v2/framepack/unload", {}),
  });
}
