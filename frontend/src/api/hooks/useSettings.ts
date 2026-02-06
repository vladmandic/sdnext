import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../client";
import type { OptionsMap, CmdFlags } from "../types/settings";

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
    },
  });
}

export function useCmdFlags() {
  return useQuery({
    queryKey: ["cmd-flags"],
    queryFn: () => api.get<CmdFlags>("/sdapi/v1/cmd-flags"),
    staleTime: Infinity,
  });
}
