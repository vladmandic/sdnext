import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../client";

export function useServerLog(lines = 200) {
  return useQuery({
    queryKey: ["server-log", lines],
    queryFn: () => api.get<string[]>("/sdapi/v1/log", { lines: String(lines) }),
    refetchInterval: 3000,
  });
}

export function useClearLog() {
  const queryClient = useQueryClient();
  return async () => {
    await api.get("/sdapi/v1/log", { clear: "true" });
    queryClient.invalidateQueries({ queryKey: ["server-log"] });
  };
}
