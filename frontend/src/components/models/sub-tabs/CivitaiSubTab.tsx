import { useState, useEffect, useCallback } from "react";
import { useCivitOptions, useCivitSearchInfinite } from "@/api/hooks/useCivitai";
import { useDownloadStore } from "@/stores/downloadStore";
import { ws, ensureWs } from "@/api/wsManager";
import type { CivitSearchParams } from "@/api/types/civitai";
import { CivitSettings } from "./civitai/CivitSettings";
import { CivitSearchHistory } from "./civitai/CivitSearchHistory";
import { CivitSearchBar } from "./civitai/CivitSearchBar";
import { CivitFilters } from "./civitai/CivitFilters";
import { CivitResultList } from "./civitai/CivitResultList";
import { CivitModelDetail } from "./civitai/CivitModelDetail";
import { CivitDownloadQueue } from "./civitai/CivitDownloadQueue";

export function CivitaiSubTab() {
  const [query, setQuery] = useState("");
  const [tag, setTag] = useState("");
  const [type, setType] = useState("");
  const [sort, setSort] = useState("");
  const [period, setPeriod] = useState("");
  const [baseModel, setBaseModel] = useState("");
  const [searchEnabled, setSearchEnabled] = useState(false);
  const [selectedModelId, setSelectedModelId] = useState<number | null>(null);

  const { data: options } = useCivitOptions();

  const searchParams: CivitSearchParams = {
    query: query || undefined,
    tag: tag || undefined,
    types: type || undefined,
    sort: sort || undefined,
    period: period || undefined,
    base_models: baseModel || undefined,
    limit: 20,
  };

  const infiniteSearch = useCivitSearchInfinite(searchParams, searchEnabled && (!!query || !!tag));

  const handleSearch = useCallback(() => {
    if (!query && !tag) return;
    if (searchEnabled) {
      infiniteSearch.refetch();
    } else {
      setSearchEnabled(true);
    }
  }, [query, tag, searchEnabled, infiniteSearch]);

  function handleHistorySelect(q: string, t: string) {
    setQuery(q);
    setTag(t);
    setSearchEnabled(false);
    // Trigger search on next tick after state updates
    setTimeout(() => setSearchEnabled(true), 0);
  }

  // WS listener for download progress
  useEffect(() => {
    ensureWs();
    const unsub = ws.on("message", (data) => {
      const msg = data as { type: string; data?: unknown };
      if (msg.type === "download" && Array.isArray(msg.data)) {
        useDownloadStore.getState().updateFromWs(msg.data);
      }
    });
    return unsub;
  }, []);

  return (
    <div className="space-y-3">
      <CivitSettings />
      <CivitSearchHistory onSelect={handleHistorySelect} />
      <CivitSearchBar query={query} tag={tag} onQueryChange={(v) => { setQuery(v); setSearchEnabled(false); }} onTagChange={(v) => { setTag(v); setSearchEnabled(false); }} onSearch={handleSearch} isLoading={infiniteSearch.isFetching} />
      <CivitFilters options={options} type={type} sort={sort} period={period} baseModel={baseModel} onTypeChange={setType} onSortChange={setSort} onPeriodChange={setPeriod} onBaseModelChange={setBaseModel} />
      <CivitResultList pages={infiniteSearch.data} hasNextPage={!!infiniteSearch.hasNextPage} isFetchingNextPage={infiniteSearch.isFetchingNextPage} fetchNextPage={() => infiniteSearch.fetchNextPage()} onSelectModel={setSelectedModelId} />
      <CivitModelDetail modelId={selectedModelId} onClose={() => setSelectedModelId(null)} />
      <CivitDownloadQueue />
    </div>
  );
}
