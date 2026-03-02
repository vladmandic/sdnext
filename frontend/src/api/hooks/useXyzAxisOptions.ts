import { useQuery, useMutation } from "@tanstack/react-query";
import { api } from "../client";

export interface XyzAxisOption {
  label: string;
  type: "int" | "float" | "str" | "str_permutations" | "bool";
  cost: number;
  category: string;
  has_choices: boolean;
  choices?: string[] | null;
}

interface XyzAxesResponse {
  items: XyzAxisOption[];
  categories: string[];
}

export interface XyzValidateResponse {
  ok: boolean;
  resolved: unknown[];
  count: number;
  errors: string[];
}

export interface XyzPreviewResponse {
  ok: boolean;
  dimensions: { x: number; y: number; z: number };
  total_cells: number;
  total_steps: number;
  execution_order: string[];
  x_values: unknown[];
  y_values: unknown[];
  z_values: unknown[];
  errors: string[];
}

export function useXyzAxisOptions() {
  return useQuery({
    queryKey: ["xyz-axis-options"],
    queryFn: async () => {
      const res = await api.get<XyzAxesResponse>("/sdapi/v2/xyz-grid/axes");
      return res.items;
    },
    staleTime: 60_000,
  });
}

export function useXyzAxisChoices(label: string, enabled: boolean) {
  return useQuery({
    queryKey: ["xyz-axis-choices", label],
    queryFn: async () => {
      const res = await api.get<XyzAxesResponse>("/sdapi/v2/xyz-grid/axes", { expand: label });
      return res.items;
    },
    enabled: enabled && label.length > 0,
    staleTime: 60_000,
  });
}

export function useXyzValidate() {
  return useMutation({
    mutationFn: (params: { axis_type: string; values: string }) =>
      api.post<XyzValidateResponse>("/sdapi/v2/xyz-grid/validate", params),
  });
}

export function useXyzPreview() {
  return useMutation({
    mutationFn: (params: {
      x_axis?: { type: string; values: string } | null;
      y_axis?: { type: string; values: string } | null;
      z_axis?: { type: string; values: string } | null;
      steps?: number;
    }) => api.post<XyzPreviewResponse>("/sdapi/v2/xyz-grid/preview", params),
  });
}
