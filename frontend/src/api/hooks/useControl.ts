import { useMutation, useQuery } from "@tanstack/react-query";
import { api } from "../client";
import type { ControlUnitType, PreprocessorInfo, PreprocessResponse } from "../types/control";
import { BACKEND_UNIT_TYPE } from "../types/control";

const TYPES_WITH_MODELS: Set<ControlUnitType> = new Set(["controlnet", "t2i", "xs", "lite", "style_transfer"]);

export function useControlModels(unitType: ControlUnitType) {
  const backendType = BACKEND_UNIT_TYPE[unitType] ?? unitType;
  return useQuery({
    queryKey: ["control-models", unitType],
    queryFn: () => api.get<string[]>(`/sdapi/v2/control-models?unit_type=${backendType}`),
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
      api.post<PreprocessResponse>("/sdapi/v2/preprocess", req),
  });
}

export function usePreprocessors() {
  return useQuery({
    queryKey: ["preprocessors"],
    queryFn: () => api.get<PreprocessorInfo[]>("/sdapi/v2/preprocessors"),
    staleTime: 60_000,
  });
}
