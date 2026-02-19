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
