import { useEffect, useRef } from "react";
import { useGenerationStore } from "@/stores/generationStore";
import { useOptionsSubset } from "@/api/hooks/useSettings";

export function useHistoryInit() {
  const hydrated = useRef(false);
  const hydrateFromDb = useGenerationStore((s) => s.hydrateFromDb);
  const setHistoryLimit = useGenerationStore((s) => s.setHistoryLimit);
  const { data: options } = useOptionsSubset(["latent_history"]);

  useEffect(() => {
    if (!hydrated.current) {
      hydrated.current = true;
      hydrateFromDb();
    }
  }, [hydrateFromDb]);

  useEffect(() => {
    if (options?.latent_history != null) {
      setHistoryLimit(Number(options.latent_history));
    }
  }, [options, setHistoryLimit]);
}
