import { useMutation, useQuery } from "@tanstack/react-query";
import { api } from "../client";
import type { ControlUnitType, PreprocessorInfo } from "../types/control";

export function useControlNetModels() {
  return useQuery({
    queryKey: ["controlnets"],
    queryFn: () => api.get<string[]>("/sdapi/v1/controlnets"),
    staleTime: 30_000,
  });
}

const TYPES_WITH_MODELS: Set<ControlUnitType> = new Set(["controlnet", "t2i", "xs", "lite", "reference"]);

export function useControlModels(unitType: ControlUnitType) {
  return useQuery({
    queryKey: ["control-models", unitType],
    queryFn: () => api.get<string[]>(`/sdapi/v2/control-models?unit_type=${unitType}`),
    staleTime: 5 * 60 * 1000,
    enabled: TYPES_WITH_MODELS.has(unitType),
  });
}

export function useControlModes() {
  return useQuery({
    queryKey: ["control-modes"],
    queryFn: () => api.get<Record<string, string[]>>("/sdapi/v2/control-modes"),
    staleTime: 5 * 60 * 1000,
  });
}

export function usePreprocessImage() {
  return useMutation({
    mutationFn: (req: { image: string; model: string; params?: Record<string, unknown> }) =>
      api.post<{ model: string; image: string }>("/sdapi/v1/preprocess", req),
  });
}

export function usePreprocessors() {
  return useQuery({
    queryKey: ["preprocessors"],
    queryFn: () => api.get<PreprocessorInfo[]>("/sdapi/v1/preprocessors"),
    staleTime: 60_000,
  });
}
