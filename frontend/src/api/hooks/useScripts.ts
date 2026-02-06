import { useQuery } from "@tanstack/react-query";
import { api } from "../client";
import type { ScriptsList, ScriptInfo } from "../types/script";

export function useScriptsList() {
  return useQuery({
    queryKey: ["scripts"],
    queryFn: () => api.get<ScriptsList>("/sdapi/v1/scripts"),
    staleTime: 60_000,
  });
}

export function useScriptInfo() {
  return useQuery({
    queryKey: ["script-info"],
    queryFn: () => api.get<ScriptInfo[]>("/sdapi/v1/script-info"),
    staleTime: 60_000,
  });
}
