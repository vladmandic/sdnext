import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../client";
import type {
  UpdateCheckResult,
  UpdateApplyRequest,
  UpdateApplyResult,
  BenchmarkRunRequest,
  BenchmarkRunResult,
  BenchmarkHistory,
  HistoryEntry,
  SystemInfoFull,
  StorageInfo,
} from "../types/system";

export function useHistory() {
  return useQuery({
    queryKey: ["history"],
    queryFn: () => api.get<HistoryEntry[]>("/sdapi/v1/history"),
    refetchInterval: 10_000,
  });
}

export function useSystemInfoFull() {
  return useQuery({
    queryKey: ["system-info-full"],
    queryFn: () => api.get<SystemInfoFull>("/sdapi/v1/system-info/status", { full: "true" }),
    staleTime: 30_000,
    enabled: false,
  });
}

export function useUpdateCheck() {
  return useQuery({
    queryKey: ["update-check"],
    queryFn: () => api.get<UpdateCheckResult>("/sdapi/v2/update/check"),
    enabled: false,
  });
}

export function useBenchmarkResults() {
  return useQuery({
    queryKey: ["benchmark-results"],
    queryFn: () => api.get<BenchmarkHistory>("/sdapi/v2/benchmark/results"),
    enabled: false,
  });
}

export function useRestartServer() {
  return useMutation({
    mutationKey: ["server-restart"],
    mutationFn: () => api.post("/sdapi/v2/server/restart"),
  });
}

export function useShutdownServer() {
  return useMutation({
    mutationKey: ["server-shutdown"],
    mutationFn: () => api.post("/sdapi/v1/shutdown"),
  });
}

export function useToggleProfiling() {
  return useMutation({
    mutationKey: ["server-profiling"],
    mutationFn: () => api.post<{ enabled: boolean }>("/sdapi/v2/server/profiling"),
  });
}

export function useApplyUpdate() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationKey: ["update-apply"],
    mutationFn: (req: UpdateApplyRequest) => api.post<UpdateApplyResult>("/sdapi/v2/update/apply", req),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["update-check"] });
      queryClient.invalidateQueries({ queryKey: ["server-info"] });
    },
  });
}

export function useStorage() {
  return useQuery({
    queryKey: ["storage"],
    queryFn: () => api.get<StorageInfo>("/sdapi/v2/storage"),
    enabled: false,
    retry: false,
  });
}

export function useRunBenchmark() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationKey: ["benchmark-run"],
    mutationFn: (req: BenchmarkRunRequest) => api.post<BenchmarkRunResult>("/sdapi/v2/benchmark/run", req),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["benchmark-results"] });
    },
  });
}
