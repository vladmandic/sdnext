import { useState, useMemo } from "react";
import { RefreshCw, ImageOff } from "lucide-react";
import { useQueryClient } from "@tanstack/react-query";
import { useExtraNetworks, usePromptStyles } from "@/api/hooks/useNetworks";
import { useSetOptions } from "@/api/hooks/useSettings";
import { useGenerationStore } from "@/stores/generationStore";
import type { LoraNetwork, PromptStyle } from "@/api/types/models";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { api } from "@/api/client";

const TYPE_FILTERS = ["All", "Checkpoint", "LoRA", "Embedding", "Style"] as const;
type TypeFilter = (typeof TYPE_FILTERS)[number];

function typeMatches(network: LoraNetwork, filter: TypeFilter): boolean {
  if (filter === "All") return true;
  const t = network.type?.toLowerCase() ?? "";
  if (filter === "Checkpoint") return t === "model" || t === "checkpoint";
  if (filter === "LoRA") return t === "lora" || t === "lycoris";
  if (filter === "Embedding") return t === "embedding" || t === "textual inversion";
  return false;
}

export function NetworksTab() {
  const { data: networks, isLoading } = useExtraNetworks();
  const { data: styles } = usePromptStyles();
  const setOptions = useSetOptions();
  const queryClient = useQueryClient();
  const [filter, setFilter] = useState<TypeFilter>("All");
  const [search, setSearch] = useState("");

  const filtered = useMemo(() => {
    const items: (LoraNetwork | PromptStyle)[] = [];

    if (filter === "All" || filter !== "Style") {
      for (const n of networks ?? []) {
        if (!typeMatches(n, filter)) continue;
        if (search && !n.name.toLowerCase().includes(search.toLowerCase())) continue;
        items.push(n);
      }
    }
    if (filter === "All" || filter === "Style") {
      for (const s of styles ?? []) {
        if (search && !s.name.toLowerCase().includes(search.toLowerCase())) continue;
        items.push(s);
      }
    }
    return items;
  }, [networks, styles, filter, search]);

  function handleClick(item: LoraNetwork | PromptStyle) {
    if ("type" in item && item.type) {
      const network = item as LoraNetwork;
      const t = network.type.toLowerCase();
      if (t === "lora" || t === "lycoris") {
        const current = useGenerationStore.getState().prompt;
        const tag = `<lora:${network.name}:1>`;
        useGenerationStore.getState().setParam("prompt", current ? `${current} ${tag}` : tag);
      } else if (t === "model" || t === "checkpoint") {
        setOptions.mutate({ sd_model_checkpoint: network.title ?? network.name });
      } else if (t === "embedding" || t === "textual inversion") {
        const current = useGenerationStore.getState().prompt;
        useGenerationStore.getState().setParam("prompt", current ? `${current} ${network.name}` : network.name);
      }
    } else {
      const style = item as PromptStyle;
      if (style.prompt) {
        const current = useGenerationStore.getState().prompt;
        useGenerationStore.getState().setParam("prompt", current ? `${current} ${style.prompt}` : style.prompt);
      }
      if (style.negative_prompt) {
        const currentNeg = useGenerationStore.getState().negativePrompt;
        useGenerationStore.getState().setParam("negativePrompt", currentNeg ? `${currentNeg} ${style.negative_prompt}` : style.negative_prompt);
      }
    }
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="p-2 space-y-2 border-b border-border">
        <div className="flex items-center gap-1.5 flex-wrap">
          {TYPE_FILTERS.map((t) => (
            <button
              key={t}
              type="button"
              onClick={() => setFilter(t)}
              className={cn(
                "px-2 py-0.5 rounded-full text-[11px] font-medium transition-colors",
                filter === t ? "bg-accent text-accent-foreground" : "bg-muted text-muted-foreground hover:text-foreground",
              )}
            >
              {t}
            </button>
          ))}
          <button
            type="button"
            onClick={() => {
              queryClient.invalidateQueries({ queryKey: ["extra-networks"] });
              queryClient.invalidateQueries({ queryKey: ["prompt-styles"] });
            }}
            className="ml-auto p-1 text-muted-foreground hover:text-foreground"
          >
            <RefreshCw className="h-3.5 w-3.5" />
          </button>
        </div>
        <Input
          placeholder="Search..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="h-7 text-xs"
        />
      </div>

      {/* Grid */}
      <div className="flex-1 overflow-auto p-2">
        {isLoading && <p className="text-xs text-muted-foreground p-2">Loading networks...</p>}
        <div className="grid grid-cols-[repeat(auto-fill,minmax(140px,1fr))] gap-1.5">
          {filtered.map((item) => (
            <NetworkCard key={item.name} item={item} onClick={() => handleClick(item)} />
          ))}
        </div>
        {!isLoading && filtered.length === 0 && (
          <p className="text-xs text-muted-foreground text-center py-6">No results found.</p>
        )}
      </div>
    </div>
  );
}

function NetworkCard({ item, onClick }: { item: LoraNetwork | PromptStyle; onClick: () => void }) {
  const isNetwork = "type" in item && item.type;
  const typeBadge = isNetwork ? (item as LoraNetwork).type : "Style";
  const preview = item.preview;
  const previewUrl = preview ? `${api.getBaseUrl()}/file=${preview}` : null;

  return (
    <button
      type="button"
      onClick={onClick}
      className="flex flex-col rounded-md border border-border overflow-hidden hover:border-accent transition-colors text-left"
    >
      <div className="aspect-square w-full bg-muted/30 flex items-center justify-center overflow-hidden">
        {previewUrl ? (
          <img src={previewUrl} alt={item.name} className="w-full h-full object-cover" loading="lazy" />
        ) : (
          <ImageOff className="h-6 w-6 text-muted-foreground/40" />
        )}
      </div>
      <div className="p-1.5 space-y-0.5">
        <p className="text-[11px] font-medium truncate leading-tight">{item.name}</p>
        <Badge variant="secondary" className="text-[9px] px-1 py-0">{typeBadge}</Badge>
      </div>
    </button>
  );
}
