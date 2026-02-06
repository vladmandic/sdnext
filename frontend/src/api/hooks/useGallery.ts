import { useQuery } from "@tanstack/react-query";
import { api } from "../client";
import type { ResHistory } from "../types/progress";

export function useHistory() {
  return useQuery({
    queryKey: ["history"],
    queryFn: () => api.get<ResHistory[]>("/sdapi/v1/history"),
    staleTime: 10_000,
  });
}
