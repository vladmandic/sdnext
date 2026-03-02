import { useMemo } from "react";
import { Loader2 } from "lucide-react";
import type { InfiniteData } from "@tanstack/react-query";
import type { CivitSearchResponse } from "@/api/types/civitai";
import { useCivitCheckLocal } from "@/api/hooks/useCivitai";
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
  const models = useMemo(() => pages?.pages.flatMap((p) => p.items) ?? [], [pages]);

  const allHashes = useMemo(() => {
    const hashes: string[] = [];
    for (const m of models) {
      for (const v of m.modelVersions) {
        for (const f of v.files) {
          if (f.hashes.SHA256) hashes.push(f.hashes.SHA256);
        }
      }
    }
    return hashes;
  }, [models]);

  const { data: localCheck } = useCivitCheckLocal(allHashes);
  const localFiles = localCheck?.found ?? {};

  if (models.length === 0) return null;

  return (
    <div className="border border-border rounded-md overflow-hidden">
      <div className="divide-y divide-border/50">
        {models.map((m) => (
          <CivitResultCard key={m.id} model={m} localFiles={localFiles} onClick={() => onSelectModel(m.id)} />
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
