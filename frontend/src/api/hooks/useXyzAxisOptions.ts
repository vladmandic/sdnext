import { useQuery } from "@tanstack/react-query";
import { api } from "../client";

export interface XyzAxisOption {
  label: string;
  type: "int" | "float" | "str" | "str_permutations" | "bool";
  cost: number;
  choices: boolean | string[];
}

export function useXyzAxisOptions() {
  return useQuery({
    queryKey: ["xyz-axis-options"],
    queryFn: () => api.get<XyzAxisOption[]>("/sdapi/v1/xyz-grid"),
    staleTime: 60_000,
  });
}

export function useXyzAxisChoices(label: string, enabled: boolean) {
  return useQuery({
    queryKey: ["xyz-axis-choices", label],
    queryFn: () => api.get<XyzAxisOption[]>("/sdapi/v1/xyz-grid", { option: label }),
    enabled: enabled && label.length > 0,
    staleTime: 60_000,
  });
}
