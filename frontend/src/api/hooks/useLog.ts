import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "../client";

interface LogResponse {
  lines: string[];
  total: number;
}

export function useServerLog(lines = 200) {
  return useQuery({
    queryKey: ["server-log", lines],
    queryFn: () => api.get<LogResponse>("/sdapi/v2/log", { lines: String(lines) }),
    select: (data) => data.lines,
    refetchInterval: 3000,
    staleTime: 3000,
  });
}

export function useClearLog() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () => api.delete("/sdapi/v2/log"),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["server-log"] });
    },
  });
}
