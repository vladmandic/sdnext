import { useQuery } from "@tanstack/react-query";
import { api } from "../client";
import type { Extension } from "../types/models";

export function useExtensions() {
  return useQuery({
    queryKey: ["extensions"],
    queryFn: () => api.get<Extension[]>("/sdapi/v2/extensions"),
    staleTime: 60_000,
  });
}
