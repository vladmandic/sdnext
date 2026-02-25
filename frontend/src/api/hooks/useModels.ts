import { useMutation, useMutationState, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../client";
import type { SdVae, Upscaler, SdModelsResponse, SamplerV2, CheckpointInfoV2 } from "../types/models";

const MODEL_MUTATION_KEY = ["model-operation"];

export function useModelList() {
  return useQuery({
    queryKey: ["models"],
    queryFn: async () => {
      const resp = await api.get<SdModelsResponse>("/sdapi/v2/sd-models");
      return resp.items;
    },
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

export function useSamplerList(modelType?: string | null) {
  const params: Record<string, string> = {};
  if (modelType) params.model_type = modelType;
  return useQuery({
    queryKey: ["samplers", modelType ?? "all"],
    queryFn: () => api.get<SamplerV2[]>("/sdapi/v2/samplers", params),
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
    queryFn: () => api.get<CheckpointInfoV2>("/sdapi/v2/checkpoint"),
    staleTime: 30_000,
  });
}

export function useLoadModel() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationKey: MODEL_MUTATION_KEY,
    mutationFn: (checkpoint: string) =>
      api.post("/sdapi/v2/checkpoint", { sd_model_checkpoint: checkpoint }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["options"] });
      queryClient.invalidateQueries({ queryKey: ["checkpoint"] });
      queryClient.invalidateQueries({ queryKey: ["samplers"] });
      queryClient.invalidateQueries({ queryKey: ["control-models"] });
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
      queryClient.invalidateQueries({ queryKey: ["samplers"] });
      queryClient.invalidateQueries({ queryKey: ["control-models"] });
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
      queryClient.invalidateQueries({ queryKey: ["samplers"] });
      queryClient.invalidateQueries({ queryKey: ["control-models"] });
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
