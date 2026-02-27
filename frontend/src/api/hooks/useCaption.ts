import { useMutation, useQuery } from "@tanstack/react-query";
import { api } from "../client";
import type {
  OpenClipRequest, OpenClipResponse,
  TaggerRequest, TaggerResponse, TaggerModel,
  VqaRequest, VqaResponse, VlmModel,
} from "../types/caption";

// ---------------------------------------------------------------------------
// OpenCLIP
// ---------------------------------------------------------------------------

export function useOpenClipModels() {
  return useQuery({
    queryKey: ["openclip-models"],
    queryFn: () => api.get<string[]>("/sdapi/v2/caption/openclip/models"),
    staleTime: 5 * 60 * 1000,
  });
}

export function useOpenClipCaption() {
  return useMutation({
    mutationFn: (params: OpenClipRequest) =>
      api.post<OpenClipResponse>("/sdapi/v2/caption/openclip", params),
  });
}

// ---------------------------------------------------------------------------
// Tagger
// ---------------------------------------------------------------------------

export function useTaggerModels() {
  return useQuery({
    queryKey: ["tagger-models"],
    queryFn: () => api.get<TaggerModel[]>("/sdapi/v2/caption/tagger/models"),
    staleTime: 5 * 60 * 1000,
  });
}

export function useTaggerCaption() {
  return useMutation({
    mutationFn: (params: TaggerRequest) =>
      api.post<TaggerResponse>("/sdapi/v2/caption/tagger", params),
  });
}

// ---------------------------------------------------------------------------
// VLM / VQA
// ---------------------------------------------------------------------------

export function useVlmModels() {
  return useQuery({
    queryKey: ["vlm-models"],
    queryFn: () => api.get<VlmModel[]>("/sdapi/v2/caption/vlm/models"),
    staleTime: 5 * 60 * 1000,
  });
}

export function useVqaCaption() {
  return useMutation({
    mutationFn: (params: VqaRequest) =>
      api.post<VqaResponse>("/sdapi/v2/caption/vlm", params),
  });
}
