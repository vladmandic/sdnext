import { useQuery } from "@tanstack/react-query";
import { api } from "../client";
import type { ResMemory, ResGPU, ServerInfo, VersionInfo, LoadedModel } from "../types/server";

async function fetchServerInfo(): Promise<ServerInfo> {
  try {
    return await api.get<ServerInfo>("/sdapi/v1/server-info");
  } catch {
    // Fall back to /version for backends without server-info endpoint
    const version = await api.get<VersionInfo>("/sdapi/v1/version");
    return {
      version: { app: version.app, updated: version.updated },
      backend: "diffusers",
      platform: "",
      gpu: "",
      api_version: "v1",
      capabilities: { txt2img: true, img2img: true, control: true, video: false, websocket: false },
      model: { name: null, type: null },
    };
  }
}

export function useServerInfo() {
  return useQuery({
    queryKey: ["server-info"],
    queryFn: fetchServerInfo,
    staleTime: 30_000,
    retry: 2,
  });
}

export function useVersion() {
  return useQuery({
    queryKey: ["version"],
    queryFn: () => api.get<VersionInfo>("/sdapi/v1/version"),
    staleTime: Infinity,
  });
}

export function useMemory() {
  return useQuery({
    queryKey: ["memory"],
    queryFn: () => api.get<ResMemory>("/sdapi/v1/memory"),
    refetchInterval: 5000,
  });
}

export function useGpuStatus() {
  return useQuery({
    queryKey: ["gpu"],
    queryFn: () => api.get<ResGPU[]>("/sdapi/v1/gpu"),
    refetchInterval: 3000,
  });
}

export function useLoadedModels() {
  return useQuery({
    queryKey: ["loaded-models"],
    queryFn: () => api.get<LoadedModel[]>("/sdapi/v1/loaded-models"),
    refetchInterval: 10_000,
    staleTime: 5_000,
  });
}
