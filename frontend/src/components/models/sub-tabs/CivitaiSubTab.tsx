import { useState, useCallback } from "react";
import { useCivitOptions, useCivitSearchInfinite, useCivitSettings, useCivitMe } from "@/api/hooks/useCivitai";
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
  const [creator, setCreator] = useState("");
  const [nsfw, setNsfw] = useState(false);
  const [favorites, setFavorites] = useState(false);
  const [searchEnabled, setSearchEnabled] = useState(false);
  const [selectedModelId, setSelectedModelId] = useState<number | null>(null);

  const { data: options } = useCivitOptions();
  const { data: settings } = useCivitSettings();
  const tokenConfigured = settings?.token_configured ?? false;
  const { data: me } = useCivitMe(tokenConfigured && favorites);

  const searchParams: CivitSearchParams = {
    query: query || undefined,
    tag: tag || undefined,
    types: type || undefined,
    sort: sort || undefined,
    period: period || undefined,
    base_models: baseModel || undefined,
    nsfw: nsfw || undefined,
    username: favorites && me?.username ? me.username : (creator || undefined),
    favorites: favorites || undefined,
    limit: 20,
  };

  const infiniteSearch = useCivitSearchInfinite(searchParams, searchEnabled && (!!query || !!tag || favorites));

  const handleSearch = useCallback(() => {
    if (!query && !tag && !favorites) return;
    if (searchEnabled) {
      infiniteSearch.refetch();
    } else {
      setSearchEnabled(true);
    }
  }, [query, tag, favorites, searchEnabled, infiniteSearch]);

  function handleHistorySelect(q: string, t: string) {
    setQuery(q);
    setTag(t);
    setSearchEnabled(false);
    setTimeout(() => setSearchEnabled(true), 0);
  }

  function handleFavoritesChange(v: boolean) {
    setFavorites(v);
    if (v) {
      setSearchEnabled(true);
    }
  }

  return (
    <div className="space-y-3">
      <CivitSettings />
      <CivitSearchHistory onSelect={handleHistorySelect} />
      <CivitSearchBar query={query} tag={tag} onQueryChange={(v) => { setQuery(v); setSearchEnabled(false); }} onTagChange={(v) => { setTag(v); setSearchEnabled(false); }} onSearch={handleSearch} isLoading={infiniteSearch.isFetching} />
      <CivitFilters options={options} type={type} sort={sort} period={period} baseModel={baseModel} creator={creator} nsfw={nsfw} favorites={favorites} tokenConfigured={tokenConfigured} onTypeChange={setType} onSortChange={setSort} onPeriodChange={setPeriod} onBaseModelChange={setBaseModel} onCreatorChange={setCreator} onNsfwChange={setNsfw} onFavoritesChange={handleFavoritesChange} />
      <CivitResultList pages={infiniteSearch.data} hasNextPage={!!infiniteSearch.hasNextPage} isFetchingNextPage={infiniteSearch.isFetchingNextPage} fetchNextPage={() => infiniteSearch.fetchNextPage()} onSelectModel={setSelectedModelId} />
      <CivitModelDetail modelId={selectedModelId} onClose={() => setSelectedModelId(null)} />
      <CivitDownloadQueue />
    </div>
  );
}
