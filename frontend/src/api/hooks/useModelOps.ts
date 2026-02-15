import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../client";
import type {
  ModelAnalysis,
  ModelListDetail,
  ModelSaveRequest,
  HfModelResult,
  HfDownloadRequest,
  CivitaiDownloadRequest,
  CivitMetadataScanResult,
  CivitMetadataUpdateResult,
  MergeMethodsInfo,
  MergeRequest,
  ReplaceRequest,
  LoaderComponentsResponse,
  LoaderLoadRequest,
  LoraExtractRequest,
} from "../types/modelOps";

// Phase 1

export function useModelAnalysis(enabled = false) {
  return useQuery({
    queryKey: ["model-analyze"],
    queryFn: () => api.get<ModelAnalysis>("/sdapi/v1/model/analyze"),
    enabled,
    staleTime: 0,
  });
}

export function useModelListDetail() {
  return useQuery({
    queryKey: ["model-list-detail"],
    queryFn: () => api.get<ModelListDetail[]>("/sdapi/v1/model/list-detail"),
    staleTime: 60_000,
  });
}

export function useSaveModel() {
  return useMutation({
    mutationFn: (req: ModelSaveRequest) => api.post<{ status: string }>("/sdapi/v1/model/save", req),
  });
}

export function useUpdateHashes() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => api.post<{ updated: Array<{ name: string; type: string; hash: string }> }>("/sdapi/v1/model/update-hashes"),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["model-list-detail"] }),
  });
}

// Phase 2

export function useHfSearch(keyword: string, enabled = false) {
  return useQuery({
    queryKey: ["hf-search", keyword],
    queryFn: () => api.get<HfModelResult[]>("/sdapi/v1/model/hf/search", { keyword }),
    enabled: enabled && keyword.length > 0,
    staleTime: 30_000,
  });
}

export function useHfDownload() {
  return useMutation({
    mutationFn: (req: HfDownloadRequest) => api.post<{ status: string }>("/sdapi/v1/model/hf/download", req),
  });
}

export function useCivitaiDownload() {
  return useMutation({
    mutationFn: (req: CivitaiDownloadRequest) => api.post<{ status: string }>("/sdapi/v1/model/civitai/download", req),
  });
}

export function useMetadataScan() {
  return useMutation({
    mutationFn: () => api.post<{ results: CivitMetadataScanResult[] }>("/sdapi/v1/model/metadata/scan"),
  });
}

export function useMetadataUpdate() {
  return useMutation({
    mutationFn: () => api.post<{ results: CivitMetadataUpdateResult[] }>("/sdapi/v1/model/metadata/update"),
  });
}

// Phase 3

export function useMergeMethods() {
  return useQuery({
    queryKey: ["merge-methods"],
    queryFn: () => api.get<MergeMethodsInfo>("/sdapi/v1/model/merge/methods"),
    staleTime: Infinity,
  });
}

export function useMergeModels() {
  return useMutation({
    mutationFn: (req: MergeRequest) => api.post<{ status: string }>("/sdapi/v1/model/merge", req),
  });
}

export function useReplaceComponents() {
  return useMutation({
    mutationFn: (req: ReplaceRequest) => api.post<{ status: string }>("/sdapi/v1/model/replace", req),
  });
}

// Phase 4

export function useLoaderPipelines() {
  return useQuery({
    queryKey: ["loader-pipelines"],
    queryFn: () => api.get<{ pipelines: string[] }>("/sdapi/v1/model/loader/pipelines"),
    staleTime: Infinity,
  });
}

export function useLoaderComponents(modelType: string, enabled = false) {
  return useQuery({
    queryKey: ["loader-components", modelType],
    queryFn: () => api.post<LoaderComponentsResponse>("/sdapi/v1/model/loader/components", { model_type: modelType }),
    enabled: enabled && modelType.length > 0,
    staleTime: 0,
  });
}

export function useLoaderLoad() {
  return useMutation({
    mutationFn: (req: LoaderLoadRequest) => api.post<{ status: string }>("/sdapi/v1/model/loader/load", req),
  });
}

export function useLoraLoaded(enabled = true) {
  return useQuery({
    queryKey: ["lora-loaded"],
    queryFn: () => api.get<{ loras: string[] }>("/sdapi/v1/model/lora/loaded"),
    enabled,
    staleTime: 30_000,
  });
}

export function useLoraExtract() {
  return useMutation({
    mutationFn: (req: LoraExtractRequest) => api.post<{ status: string }>("/sdapi/v1/model/lora/extract", req),
  });
}
