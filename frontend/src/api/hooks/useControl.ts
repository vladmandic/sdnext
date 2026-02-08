import { useQuery } from "@tanstack/react-query";
import { api } from "../client";
import type { ControlUnitType, PreprocessorInfo } from "../types/control";

export function useControlNetModels() {
  return useQuery({
    queryKey: ["controlnets"],
    queryFn: () => api.get<string[]>("/sdapi/v1/controlnets"),
    staleTime: 30_000,
  });
}

export function useControlModels(unitType: ControlUnitType) {
  return useQuery({
    queryKey: ["control-models", unitType],
    queryFn: () => api.get<string[]>(`/sdapi/v1/control-models?unit_type=${unitType}`),
    staleTime: 5 * 60 * 1000,
  });
}

export function usePreprocessors() {
  return useQuery({
    queryKey: ["preprocessors"],
    queryFn: () => api.get<PreprocessorInfo[]>("/sdapi/v1/preprocessors"),
    staleTime: 60_000,
  });
}
