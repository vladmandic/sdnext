import { useMutation } from "@tanstack/react-query";
import { api } from "../client";

interface PngInfoResponse {
  ok: boolean;
  info: string;
  items: Record<string, string>;
  parameters: Record<string, unknown>;
}

export function usePngInfo() {
  return useMutation({
    mutationFn: (params: { image: string }) =>
      api.post<PngInfoResponse>("/sdapi/v2/png-info", params),
  });
}
