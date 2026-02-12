import { useMutation, useMutationState, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../client";
import type { SdModel, SdVae, Sampler, Upscaler } from "../types/models";

const MODEL_MUTATION_KEY = ["model-operation"];

export interface CheckpointInfo {
  type: string | null;
  class: string | null;
  title?: string;
  name?: string;
  filename?: string;
  hash?: string;
}

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

export function useCurrentCheckpoint() {
  return useQuery({
    queryKey: ["checkpoint"],
    queryFn: () => api.get<CheckpointInfo>("/sdapi/v1/checkpoint"),
    staleTime: 30_000,
  });
}

export function useLoadModel() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationKey: MODEL_MUTATION_KEY,
    mutationFn: (checkpoint: string) =>
      api.post(`/sdapi/v1/checkpoint?sd_model_checkpoint=${encodeURIComponent(checkpoint)}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["options"] });
      queryClient.invalidateQueries({ queryKey: ["checkpoint"] });
    },
  });
}

export function useRefreshModels() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationKey: MODEL_MUTATION_KEY,
    mutationFn: () => api.post("/sdapi/v1/refresh-checkpoints"),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["models"] });
      queryClient.invalidateQueries({ queryKey: ["checkpoint"] });
    },
  });
}

export function useReloadModel() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationKey: MODEL_MUTATION_KEY,
    mutationFn: () => api.post("/sdapi/v1/reload-checkpoint"),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["options"] });
      queryClient.invalidateQueries({ queryKey: ["checkpoint"] });
    },
  });
}

export function useUnloadModel() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationKey: MODEL_MUTATION_KEY,
    mutationFn: () => api.post("/sdapi/v1/unload-checkpoint"),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["options"] });
      queryClient.invalidateQueries({ queryKey: ["checkpoint"] });
    },
  });
}

export function useIsModelLoading() {
  const pending = useMutationState({
    filters: { mutationKey: MODEL_MUTATION_KEY, status: "pending" },
    select: (m) => m.state.status,
  });
  return pending.length > 0;
}
