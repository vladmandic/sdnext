import { Loader2 } from "lucide-react";
import type { InfiniteData } from "@tanstack/react-query";
import type { CivitSearchResponse } from "@/api/types/civitai";
import { Button } from "@/components/ui/button";
import { CivitResultCard } from "./CivitResultCard";

interface CivitResultListProps {
  pages: InfiniteData<CivitSearchResponse> | undefined;
  hasNextPage: boolean;
  isFetchingNextPage: boolean;
  fetchNextPage: () => void;
  onSelectModel: (modelId: number) => void;
}

export function CivitResultList({ pages, hasNextPage, isFetchingNextPage, fetchNextPage, onSelectModel }: CivitResultListProps) {
  const models = pages?.pages.flatMap((p) => p.items) ?? [];

  if (models.length === 0) return null;

  return (
    <div className="border border-border rounded-md">
      <div className="divide-y divide-border/50">
        {models.map((m) => (
          <CivitResultCard key={m.id} model={m} onClick={() => onSelectModel(m.id)} />
        ))}
      </div>
      {hasNextPage && (
        <div className="p-2 flex justify-center border-t border-border/50">
          <Button size="sm" variant="ghost" onClick={fetchNextPage} disabled={isFetchingNextPage} className="text-xs">
            {isFetchingNextPage && <Loader2 className="h-3 w-3 animate-spin mr-1" />}
            Load more
          </Button>
        </div>
      )}
    </div>
  );
}
