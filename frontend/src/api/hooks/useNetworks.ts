import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "../client";
import type { PromptStyleV2, NetworkDetail, ExtraNetworksResponse } from "../types/models";

interface RefreshNetworksResponse {
  ok: boolean;
  total: number;
}

export function useExtraNetworks(params: { page?: string; search?: string; subfolder?: string; offset?: number; limit?: number } = {}) {
  const queryParams: Record<string, string> = {};
  if (params.page) queryParams.page = params.page;
  if (params.search) queryParams.search = params.search;
  if (params.subfolder) queryParams.subfolder = params.subfolder;
  if (params.offset != null) queryParams.offset = String(params.offset);
  if (params.limit != null) queryParams.limit = String(params.limit);
  return useQuery({
    queryKey: ["extra-networks", params],
    queryFn: () => api.get<ExtraNetworksResponse>("/sdapi/v2/extra-networks", queryParams),
    staleTime: 60_000,
  });
}

export function usePromptStyles() {
  return useQuery({
    queryKey: ["prompt-styles"],
    queryFn: () => api.get<PromptStyleV2[]>("/sdapi/v2/prompt-styles"),
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
    mutationFn: () => api.post<RefreshNetworksResponse>("/sdapi/v2/extra-networks/refresh"),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["extra-networks"] });
      queryClient.invalidateQueries({ queryKey: ["prompt-styles"] });
    },
  });
}
