import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/client";
import { useLoadedModels } from "@/api/hooks/useServer";
import type { SdModel } from "@/api/types/models";
import { ScrollArea } from "@/components/ui/scroll-area";

export function ModelsTab() {
  const { data: models } = useQuery({
    queryKey: ["sd-models"],
    queryFn: () => api.get<SdModel[]>("/sdapi/v1/sd-models"),
    staleTime: 60_000,
  });
  const { data: loaded } = useLoadedModels();
  const loadedCount = loaded?.length ?? 0;

  return (
    <div className="p-3 space-y-3">
      <p className="text-xs text-muted-foreground">
        {loadedCount} model{loadedCount !== 1 ? "s" : ""} loaded &middot; {models?.length ?? 0} available
      </p>
      <ScrollArea className="max-h-[60vh]">
        <div className="space-y-1">
          {models?.map((m) => (
            <div key={m.title} className="text-xs py-1 px-2 rounded hover:bg-muted/50 truncate font-mono">
              {m.model_name}
            </div>
          ))}
        </div>
      </ScrollArea>
    </div>
  );
}
