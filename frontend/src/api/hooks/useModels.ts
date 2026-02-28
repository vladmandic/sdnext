import { useMemo } from "react";
import { useMutation, useMutationState, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../client";
import type { VaeV2, UpscalerV2, SdModelsResponse, SamplerV2, CheckpointInfoV2 } from "../types/models";
import type { ComboboxGroup } from "@/components/ui/combobox";

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
    queryFn: () => api.get<VaeV2[]>("/sdapi/v2/sd-vae"),
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
    queryFn: () => api.get<UpscalerV2[]>("/sdapi/v2/upscalers"),
    staleTime: 300_000,
  });
}

const UPSCALER_GROUP_ORDER = [
  "None", "Resize", "Latent", "VIPS", "DCC Interpolation", "HQX", "ICB",
  "ESRGAN", "RealESRGAN", "SwinIR", "SCUNet", "Spandrel", "chaiNNer",
  "Diffusion", "Asymmetric VAE", "WAN", "Aura SR", "SeedVR2",
];

function stripGroupPrefix(name: string, group: string): string {
  if (name === group || name === "None") return name;
  if (name.startsWith(group + " ")) return name.slice(group.length + 1);
  return name;
}

export function useUpscalerGroups(opts?: { excludeLatent?: boolean }) {
  const { data: upscalers } = useUpscalerList();
  return useMemo<ComboboxGroup[]>(() => {
    if (!upscalers?.length) return [];
    const buckets: Record<string, { value: string; label: string }[]> = {};
    for (const u of upscalers) {
      if (opts?.excludeLatent && u.name.startsWith("Latent")) continue;
      const g = u.group || "Other";
      (buckets[g] ??= []).push({ value: u.name, label: stripGroupPrefix(u.name, g) });
    }
    const ordered = UPSCALER_GROUP_ORDER.filter((g) => buckets[g]?.length);
    const extra = Object.keys(buckets).filter((g) => !UPSCALER_GROUP_ORDER.includes(g));
    return [...ordered, ...extra].map((g) => ({ heading: g, options: buckets[g] }));
  }, [upscalers, opts?.excludeLatent]);
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
    mutationFn: () => api.post("/sdapi/v2/checkpoint/refresh"),
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
    mutationFn: () => api.post("/sdapi/v2/checkpoint/reload"),
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
    mutationFn: () => api.post("/sdapi/v2/checkpoint/unload"),
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
