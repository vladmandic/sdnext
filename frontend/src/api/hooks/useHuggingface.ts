import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "../client";

export interface HfSettings {
  token_configured: boolean;
}

export interface HfProfile {
  username: string;
  fullname: string;
  avatar: string;
}

export function useHfSettings() {
  return useQuery({
    queryKey: ["hf-settings"],
    queryFn: () => api.get<HfSettings>("/sdapi/v2/huggingface/settings"),
    staleTime: 60_000,
  });
}

export function useHfSaveSettings() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (req: { token: string }) => api.post<HfSettings>("/sdapi/v2/huggingface/settings", req),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["hf-settings"] });
      qc.invalidateQueries({ queryKey: ["hf-me"] });
      qc.invalidateQueries({ queryKey: ["secrets-status"] });
    },
  });
}

export function useHfMe(enabled = true) {
  return useQuery({
    queryKey: ["hf-me"],
    queryFn: () => api.get<HfProfile>("/sdapi/v2/huggingface/me"),
    enabled,
    staleTime: 300_000,
    retry: false,
  });
}
