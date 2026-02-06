import { useQuery } from "@tanstack/react-query";
import { api } from "../client";
import type { DetailerModel } from "../types/detailer";

export function useDetailerModels() {
  return useQuery({
    queryKey: ["detailers"],
    queryFn: () => api.get<DetailerModel[]>("/sdapi/v1/detailers"),
    staleTime: 60_000,
  });
}
