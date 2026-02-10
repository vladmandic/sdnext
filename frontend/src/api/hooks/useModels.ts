import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../client";
import type { SdModel, SdVae, Sampler, Upscaler } from "../types/models";

export function useModelList() {
  return useQuery({
    queryKey: ["models"],
    queryFn: () => api.get<SdModel[]>("/sdapi/v1/sd-models"),
    staleTime: 60_000,
  });
}

export function useVaeList() {
  return useQuery({
    queryKey: ["vaes"],
    queryFn: () => api.get<SdVae[]>("/sdapi/v1/sd-vae"),
    staleTime: 60_000,
  });
}

export function useSamplerList() {
  return useQuery({
    queryKey: ["samplers"],
    queryFn: () => api.get<Sampler[]>("/sdapi/v1/samplers"),
    staleTime: 300_000,
  });
}

export function useUpscalerList() {
  return useQuery({
    queryKey: ["upscalers"],
    queryFn: () => api.get<Upscaler[]>("/sdapi/v1/upscalers"),
    staleTime: 300_000,
  });
}

export function useLoadModel() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (checkpoint: string) =>
      api.post(`/sdapi/v1/checkpoint?sd_model_checkpoint=${encodeURIComponent(checkpoint)}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["options"] });
    },
  });
}

export function useRefreshModels() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () => api.post("/sdapi/v1/refresh-checkpoints"),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["models"] });
    },
  });
}
