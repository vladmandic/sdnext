import { useQuery, useInfiniteQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "../client";
import type { CivitOptions, CivitSearchResponse, CivitModel, CivitSearchParams, CivitDownloadRequest, CivitDownloadItem, CivitDownloadStatus, CivitHistoryEntry, CivitSettings, CivitSettingsUpdate } from "../types/civitai";

function buildSearchParams(p: CivitSearchParams): Record<string, string> {
  const out: Record<string, string> = {};
  if (p.query) out.query = p.query;
  if (p.tag) out.tag = p.tag;
  if (p.types) out.types = p.types;
  if (p.sort) out.sort = p.sort;
  if (p.period) out.period = p.period;
  if (p.base_models) out.base_models = p.base_models;
  if (p.nsfw !== undefined) out.nsfw = String(p.nsfw);
  if (p.limit) out.limit = String(p.limit);
  if (p.cursor) out.cursor = p.cursor;
  return out;
}

export function useCivitOptions() {
  return useQuery({
    queryKey: ["civitai-options"],
    queryFn: () => api.get<CivitOptions>("/sdapi/v2/civitai/options"),
    staleTime: Infinity,
  });
}

export function useCivitSearch(params: CivitSearchParams, enabled = false) {
  return useQuery({
    queryKey: ["civitai-search", params],
    queryFn: () => api.get<CivitSearchResponse>("/sdapi/v2/civitai/search", buildSearchParams(params)),
    enabled,
    staleTime: 30_000,
  });
}

export function useCivitSearchInfinite(params: CivitSearchParams, enabled = false) {
  return useInfiniteQuery({
    queryKey: ["civitai-search-infinite", params],
    queryFn: ({ pageParam }) => {
      const p = { ...params };
      if (pageParam) p.cursor = pageParam as string;
      return api.get<CivitSearchResponse>("/sdapi/v2/civitai/search", buildSearchParams(p));
    },
    initialPageParam: undefined as string | undefined,
    getNextPageParam: (lastPage) => lastPage.metadata?.nextCursor ?? undefined,
    enabled,
    staleTime: 30_000,
  });
}

export function useCivitModel(modelId: number | null, enabled = true) {
  return useQuery({
    queryKey: ["civitai-model", modelId],
    queryFn: () => api.get<CivitModel>(`/sdapi/v2/civitai/model/${modelId}`),
    enabled: enabled && modelId !== null,
    staleTime: 60_000,
  });
}

export function useCivitDownload() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (req: CivitDownloadRequest) => api.post<CivitDownloadItem>("/sdapi/v2/civitai/download", req),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["civitai-download-status"] }),
  });
}

export function useCivitDownloadCancel() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (downloadId: string) => api.post<{ success: boolean; id: string }>(`/sdapi/v2/civitai/download/${downloadId}/cancel`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["civitai-download-status"] }),
  });
}

export function useCivitDownloadStatus(enabled = false) {
  return useQuery({
    queryKey: ["civitai-download-status"],
    queryFn: () => api.get<CivitDownloadStatus>("/sdapi/v2/civitai/download/status"),
    enabled,
    refetchInterval: enabled ? 5_000 : false,
  });
}

export function useCivitHistory() {
  return useQuery({
    queryKey: ["civitai-history"],
    queryFn: () => api.get<{ history: CivitHistoryEntry[] }>("/sdapi/v2/civitai/history"),
    staleTime: 30_000,
  });
}

export function useCivitClearHistory() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => api.delete<{ success: boolean }>("/sdapi/v2/civitai/history"),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["civitai-history"] }),
  });
}

export function useCivitSettings() {
  return useQuery({
    queryKey: ["civitai-settings"],
    queryFn: () => api.get<CivitSettings>("/sdapi/v2/civitai/settings"),
    staleTime: 60_000,
  });
}

export function useCivitSaveSettings() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (req: CivitSettingsUpdate) => api.post<CivitSettings>("/sdapi/v2/civitai/settings", req),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["civitai-settings"] }),
  });
}

export function useCivitResolvePath(params: Record<string, string>, enabled = false) {
  return useQuery({
    queryKey: ["civitai-resolve-path", params],
    queryFn: () => api.get<{ path: string }>("/sdapi/v2/civitai/resolve-path", params),
    enabled,
    staleTime: 0,
  });
}
