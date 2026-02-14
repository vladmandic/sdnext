import { useQuery } from "@tanstack/react-query";
import { api } from "../client";
import type { LoraNetwork, EmbeddingsResponse, PromptStyle, NetworkDetail } from "../types/models";

export function useExtraNetworks() {
  return useQuery({
    queryKey: ["extra-networks"],
    queryFn: () => api.get<LoraNetwork[]>("/sdapi/v1/extra-networks"),
    staleTime: 60_000,
  });
}

export function useEmbeddings() {
  return useQuery({
    queryKey: ["embeddings"],
    queryFn: () => api.get<EmbeddingsResponse>("/sdapi/v1/embeddings"),
    staleTime: 60_000,
  });
}

export function usePromptStyles() {
  return useQuery({
    queryKey: ["prompt-styles"],
    queryFn: () => api.get<PromptStyle[]>("/sdapi/v1/prompt-styles"),
    staleTime: 60_000,
  });
}

export function useNetworkDetail(page: string, name: string, enabled: boolean) {
  return useQuery({
    queryKey: ["network-detail", page, name],
    queryFn: () => api.get<NetworkDetail>("/sdapi/v1/extra-networks/detail", { page, name }),
    enabled,
    staleTime: 60_000,
  });
}
