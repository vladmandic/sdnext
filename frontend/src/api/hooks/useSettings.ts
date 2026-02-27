import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../client";
import type { OptionsMap, OptionsInfoResponse, SetOptionsResponse } from "../types/settings";

export function useOptions() {
  return useQuery({
    queryKey: ["options"],
    queryFn: () => api.get<OptionsMap>("/sdapi/v2/options"),
    staleTime: 30_000,
  });
}

export function useOptionsSubset(keys: string[]) {
  const keysStr = keys.join(",");
  return useQuery({
    queryKey: ["options", keysStr],
    queryFn: () => api.get<OptionsMap>("/sdapi/v2/options", { keys: keysStr }),
    staleTime: 30_000,
    enabled: keys.length > 0,
  });
}

export function useSetOptions() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (options: Partial<OptionsMap>) =>
      api.post<SetOptionsResponse>("/sdapi/v2/options", options),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["options"] });
      queryClient.invalidateQueries({ queryKey: ["options-info"] });
    },
  });
}

export function useOptionsInfo() {
  return useQuery({
    queryKey: ["options-info"],
    queryFn: () => api.get<OptionsInfoResponse>("/sdapi/v2/options-info"),
    staleTime: 5 * 60_000,
  });
}
