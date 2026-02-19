import { useQuery } from "@tanstack/react-query";
import type { ResStatus } from "../types/progress";

export function useStatus() {
  return useQuery({
    queryKey: ["status"],
    queryFn: async () => {
      const res = await fetch("/sdapi/v1/status");
      return (await res.json()) as ResStatus;
    },
    refetchInterval: 2000,
  });
}
