import { useQuery } from "@tanstack/react-query";
import { api } from "../client";
import type { PreprocessorInfo } from "../types/control";

export function useControlNetModels() {
  return useQuery({
    queryKey: ["controlnets"],
    queryFn: () => api.get<string[]>("/sdapi/v1/controlnets"),
    staleTime: 30_000,
  });
}

export function usePreprocessors() {
  return useQuery({
    queryKey: ["preprocessors"],
    queryFn: () => api.get<PreprocessorInfo[]>("/sdapi/v1/preprocessors"),
    staleTime: 60_000,
  });
}
