import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../client";
import type { Job, JobListResponse, JobRequest } from "../types/v2";

export function useSubmitJob() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (params: JobRequest) => api.post<Job>("/sdapi/v2/jobs", params),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["v2-jobs"] }),
  });
}

export function useJob(jobId: string | null) {
  return useQuery({
    queryKey: ["v2-job", jobId],
    queryFn: () => api.get<Job>(`/sdapi/v2/jobs/${jobId}`),
    enabled: !!jobId,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      if (!status || status === "pending" || status === "running") return 1000;
      return false;
    },
  });
}

export function useJobList(params?: {
  status?: string;
  type?: string;
  limit?: number;
  offset?: number;
}) {
  const queryParams: Record<string, string> = {};
  if (params?.status) queryParams.status = params.status;
  if (params?.type) queryParams.type = params.type;
  if (params?.limit !== undefined) queryParams.limit = String(params.limit);
  if (params?.offset !== undefined) queryParams.offset = String(params.offset);

  return useQuery({
    queryKey: ["v2-jobs", params],
    queryFn: () => api.get<JobListResponse>("/sdapi/v2/jobs", queryParams),
    staleTime: 5000,
  });
}

export function useCancelJob() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (jobId: string) => api.delete(`/sdapi/v2/jobs/${jobId}`),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["v2-jobs"] }),
  });
}
