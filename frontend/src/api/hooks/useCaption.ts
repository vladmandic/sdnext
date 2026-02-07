import { useMutation, useQuery } from "@tanstack/react-query";
import { api } from "../client";
import type { InterrogateRequest, InterrogateResponse, VqaRequest, VqaResponse, VlmOptions, OpenClipOptions, TaggerOptions } from "../types/caption";

export function useInterrogateModels() {
  return useQuery({
    queryKey: ["interrogate-models"],
    queryFn: () => api.get<string[]>("/sdapi/v1/interrogate"),
    staleTime: 5 * 60 * 1000,
  });
}

export function useInterrogate() {
  return useMutation({
    mutationFn: (params: InterrogateRequest) =>
      api.post<InterrogateResponse>("/sdapi/v1/interrogate", params),
  });
}

export function useVqa() {
  return useMutation({
    mutationFn: (params: VqaRequest) =>
      api.post<VqaResponse>("/sdapi/v1/vqa", params),
  });
}

/** Set server-side caption options before calling the caption endpoint */
export function setCaptionOptions(options: VlmOptions | OpenClipOptions | TaggerOptions) {
  return api.post("/sdapi/v1/options", options);
}
