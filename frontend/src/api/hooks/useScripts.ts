import { useQuery } from "@tanstack/react-query";
import { api } from "../client";
import type { ScriptsResponse } from "../types/script";

export function useScripts() {
  return useQuery({
    queryKey: ["scripts"],
    queryFn: () => api.get<ScriptsResponse>("/sdapi/v2/scripts"),
    staleTime: 60_000,
  });
}
