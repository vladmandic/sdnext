import { useQuery, useMutation } from "@tanstack/react-query";
import { api } from "../client";
import type { PromptEnhanceModel, PromptEnhanceRequest, PromptEnhanceResponse } from "../types/promptEnhance";

export function usePromptEnhanceModels() {
  return useQuery({
    queryKey: ["prompt-enhance-models"],
    queryFn: () => api.get<PromptEnhanceModel[]>("/sdapi/v1/prompt-enhance/models"),
    staleTime: 5 * 60 * 1000,
  });
}

export function usePromptEnhance() {
  return useMutation({
    mutationFn: (params: PromptEnhanceRequest) =>
      api.post<PromptEnhanceResponse>("/sdapi/v1/prompt-enhance", params),
  });
}
