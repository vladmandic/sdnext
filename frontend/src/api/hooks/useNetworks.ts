import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "../client";
import type { LoraNetwork, EmbeddingsResponse, PromptStyle, NetworkDetail, NetworkDetailsResponse } from "../types/models";

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
    queryFn: () => api.get<NetworkDetail>("/sdapi/v2/extra-networks/detail", { page, name }),
    enabled,
    staleTime: 60_000,
  });
}

export function useRefreshNetworks() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () => api.post("/sdapi/v1/refresh-loras"),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["extra-networks"] });
      queryClient.invalidateQueries({ queryKey: ["prompt-styles"] });
    },
  });
}

export function useNetworkDetails(params: { page?: string; name?: string; filename?: string; title?: string; fullname?: string; hash?: string; offset?: number; limit?: number } = {}) {
  const queryParams: Record<string, string> = {};
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined) queryParams[k] = String(v);
  }
  return useQuery({
    queryKey: ["network-details", params],
    queryFn: () => api.get<NetworkDetailsResponse>("/sdapi/v2/extra-networks/details", queryParams),
    staleTime: 60_000,
  });
}
