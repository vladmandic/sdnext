import { useEffect, useRef } from "react";
import { toast } from "sonner";
import { useCurrentCheckpoint } from "@/api/hooks/useModels";
import { useGenerationStore } from "@/stores/generationStore";
import { useUiStore } from "@/stores/uiStore";
import { getModelDefaults, formatSuggestion } from "@/lib/modelDefaults";

export function useModelDefaultsSuggester() {
  const { data: checkpoint } = useCurrentCheckpoint();
  const prevType = useRef<string | null | undefined>(undefined);

  useEffect(() => {
    const type = checkpoint?.type ?? null;

    // Skip the very first render (initial load) and only react to changes
    if (prevType.current === undefined) {
      prevType.current = type;
      return;
    }

    // No change
    if (type === prevType.current) return;
    prevType.current = type;

    const defaults = getModelDefaults(type);
    if (!defaults) return;

    const modelName = checkpoint?.name ?? checkpoint?.title ?? type ?? "model";
    const summary = formatSuggestion(defaults);

    const autoApply = useUiStore.getState().autoApplyModelDefaults;
    if (autoApply) {
      useGenerationStore.getState().setParams(defaults);
      toast.success(`Applied defaults for ${modelName}: ${summary}`);
    } else {
      toast(`Loaded ${modelName}`, {
        description: `Suggested: ${summary}`,
        action: {
          label: "Apply",
          onClick: () => {
            useGenerationStore.getState().setParams(defaults);
            toast.success(`Applied defaults for ${modelName}`);
          },
        },
        duration: 8000,
      });
    }
  }, [checkpoint?.type, checkpoint?.name, checkpoint?.title]);
}
