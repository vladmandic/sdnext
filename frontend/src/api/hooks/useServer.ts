import { useQuery } from "@tanstack/react-query";
import { api } from "../client";
import type { ResMemory, ResGPU, ServerInfo, LoadedModel } from "../types/server";

export function useServerInfo() {
  return useQuery({
    queryKey: ["server-info"],
    queryFn: () => api.get<ServerInfo>("/sdapi/v2/server-info"),
    staleTime: 30_000,
    retry: 2,
  });
}

export function useMemory() {
  return useQuery({
    queryKey: ["memory"],
    queryFn: () => api.get<ResMemory>("/sdapi/v2/memory"),
    refetchInterval: 5000,
    staleTime: 5000,
  });
}

export function useGpuStatus() {
  return useQuery({
    queryKey: ["gpu"],
    queryFn: () => api.get<ResGPU[]>("/sdapi/v2/gpu"),
    refetchInterval: 3000,
    staleTime: 3000,
  });
}

export function useCapabilities() {
  const { data } = useServerInfo();
  return data?.capabilities ?? null;
}

export function useLoadedModels() {
  return useQuery({
    queryKey: ["loaded-models"],
    queryFn: () => api.get<LoadedModel[]>("/sdapi/v2/loaded-models"),
    refetchInterval: 10_000,
    staleTime: 5_000,
  });
}
