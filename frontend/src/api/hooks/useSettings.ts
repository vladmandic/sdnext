import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../client";
import type { OptionsMap, OptionsInfoResponse } from "../types/settings";

export function useOptions() {
  return useQuery({
    queryKey: ["options"],
    queryFn: () => api.get<OptionsMap>("/sdapi/v1/options"),
    staleTime: 30_000,
  });
}

export function useSetOptions() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (options: Partial<OptionsMap>) =>
      api.post("/sdapi/v1/options", options),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["options"] });
      queryClient.invalidateQueries({ queryKey: ["options-info"] });
    },
  });
}

export function useOptionsInfo() {
  return useQuery({
    queryKey: ["options-info"],
    queryFn: () => api.get<OptionsInfoResponse>("/sdapi/v1/options-info"),
    staleTime: 5 * 60_000,
  });
}

