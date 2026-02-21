import { useQuery } from "@tanstack/react-query";
import { api } from "../client";

export function useIPAdapterModels() {
  return useQuery({
    queryKey: ["ip-adapters"],
    queryFn: () => api.get<string[]>("/sdapi/v2/ip-adapters"),
    staleTime: 30_000,
  });
}
