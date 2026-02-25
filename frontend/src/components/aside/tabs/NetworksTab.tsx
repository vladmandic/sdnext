import { useState, useMemo } from "react";
import { RefreshCw, ImageOff, Info, Loader2 } from "lucide-react";
import { useExtraNetworks, usePromptStyles, useNetworkDetail, useRefreshNetworks } from "@/api/hooks/useNetworks";
import { useOptions, useSetOptions } from "@/api/hooks/useSettings";
import { useGenerationStore } from "@/stores/generationStore";
import type { ExtraNetworkV2, PromptStyle } from "@/api/types/models";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Popover, PopoverTrigger, PopoverContent } from "@/components/ui/popover";
import { cn } from "@/lib/utils";
import { api } from "@/api/client";

const TYPE_FILTERS = ["Model", "LoRA", "Style", "Wildcards", "Embedding", "VAE"] as const;
type TypeFilter = (typeof TYPE_FILTERS)[number];

const PAGE_MAP: Record<TypeFilter, string | null> = {
  Model: "model",
  LoRA: "lora",
  Style: null,
  Wildcards: "wildcards",
  Embedding: "embedding",
  VAE: "vae",
};

const TAG_CATEGORIES = ["Distilled", "Quantized", "Nunchaku", "Community", "Cloud"] as const;

const EXCLUDED_VERSIONS = new Set(["ref", "reference", "ready", "download"]);

type SidebarGroup = { header?: string; items: string[] };

function isExtraNetwork(item: ExtraNetworkV2 | PromptStyle): item is ExtraNetworkV2 {
  return "type" in item && !!item.type;
}

function isReferenceName(name: string): boolean {
  return name.toLowerCase().includes("reference/");
}

function isItemActive(item: ExtraNetworkV2 | PromptStyle, prompt: string, options: Record<string, unknown> | undefined): boolean {
  if (!isExtraNetwork(item)) {
    const style = item as PromptStyle;
    return !!style.prompt && prompt.includes(style.prompt);
  }
  const t = item.type?.toLowerCase() ?? "";
  if (t === "model" || t === "checkpoint") return (item.title ?? item.name) === (options?.sd_model_checkpoint as string);
  if (t === "lora" || t === "lycoris") return prompt.includes(`<lora:${item.name}:`);
  if (t === "embedding" || t === "textual inversion") return prompt.includes(item.name);
  if (t === "wildcards") return prompt.includes(`__${item.name}__`);
  if (t === "vae") return (item.title ?? item.name) === (options?.sd_vae as string);
  return false;
}

function itemHasTag(item: ExtraNetworkV2, tag: string): boolean {
  return item.tags.some((t) => t.toLowerCase() === tag.toLowerCase());
}

export function NetworksTab() {
  const { data: options } = useOptions();
  const setOptions = useSetOptions();
  const { data: styles } = usePromptStyles();
  const prompt = useGenerationStore((s) => s.prompt);
  const refreshNetworks = useRefreshNetworks();
  const [filter, setFilter] = useState<TypeFilter>("Model");
  const [search, setSearch] = useState("");
  const [selectedSubfolder, setSelectedSubfolder] = useState("All");

  const page = PAGE_MAP[filter];
  const { data: networksResp, isLoading } = useExtraNetworks({
    page: page ?? undefined,
    search: search || undefined,
    limit: 500,
  });
  const networks = networksResp?.items;

  const filtered = useMemo(() => {
    const items: (ExtraNetworkV2 | PromptStyle)[] = [];
    if (filter === "Style") {
      const lowerSearch = search.toLowerCase();
      for (const s of styles ?? []) {
        if (lowerSearch && !s.name.toLowerCase().includes(lowerSearch)) continue;
        items.push(s);
      }
    } else {
      for (const n of networks ?? []) {
        items.push(n);
      }
    }
    return items;
  }, [networks, styles, filter, search]);

  const versionSet = useMemo(() => {
    const versions = new Set<string>();
    for (const item of filtered) {
      if (isExtraNetwork(item) && item.version && !EXCLUDED_VERSIONS.has(item.version.toLowerCase())) {
        versions.add(item.version);
      }
    }
    return versions;
  }, [filtered]);

  const sidebarGroups = useMemo((): SidebarGroup[] => {
    const sortedVersions = Array.from(versionSet).sort((a, b) => a.localeCompare(b, undefined, { sensitivity: "base" }));

    if (filter === "Model") {
      const realDirs = new Set<string>();
      let hasLocal = false;
      let hasDiffusers = false;
      let hasReference = false;
      const tagHits = new Map<string, boolean>();
      for (const cat of TAG_CATEGORIES) tagHits.set(cat.toLowerCase(), false);

      for (const item of filtered) {
        if (!isExtraNetwork(item)) continue;
        const isRef = isReferenceName(item.name);
        const isDiff = item.name.startsWith("Diffusers/");
        if (!isRef && !isDiff) {
          hasLocal = true;
          const name = item.name.startsWith("models/") ? item.name.substring(7) : item.name;
          const slash = name.indexOf("/");
          if (slash > 0) realDirs.add(name.substring(0, slash));
        }
        if (isDiff) hasDiffusers = true;
        if (isRef && item.tags.length === 0) hasReference = true;
        for (const cat of TAG_CATEGORIES) {
          if (itemHasTag(item, cat.toLowerCase())) tagHits.set(cat.toLowerCase(), true);
        }
      }

      const categories: string[] = [];
      if (hasLocal) categories.push("Local");
      if (hasDiffusers) categories.push("Diffusers");
      if (hasReference) categories.push("Reference");
      for (const cat of TAG_CATEGORIES) {
        if (tagHits.get(cat.toLowerCase())) categories.push(cat);
      }
      const dirs = Array.from(realDirs).sort((a, b) => a.localeCompare(b, undefined, { sensitivity: "base" }));

      const groups: SidebarGroup[] = [{ items: ["All", ...categories] }];
      if (sortedVersions.length > 0) groups.push({ header: "Class", items: sortedVersions });
      if (dirs.length > 0) groups.push({ header: "Folders", items: dirs });
      return groups;
    }

    if (filter === "Style") {
      const categories: string[] = [];
      for (const item of filtered) {
        if (isReferenceName(item.name)) { if (!categories.includes("Reference")) categories.push("Reference"); }
        else { if (!categories.includes("Local")) categories.unshift("Local"); }
      }
      return [{ items: ["All", ...categories] }];
    }

    // LoRA, Wildcards, Embedding, VAE
    const dirs = new Set<string>();
    for (const item of filtered) {
      const slash = item.name.indexOf("/");
      if (slash > 0) dirs.add(item.name.substring(0, slash));
    }
    const sortedDirs = Array.from(dirs).sort((a, b) => a.localeCompare(b, undefined, { sensitivity: "base" }));

    const groups: SidebarGroup[] = [{ items: ["All", ...sortedDirs] }];
    if (sortedVersions.length > 0) groups.push({ header: "Class", items: sortedVersions });
    return groups;
  }, [filtered, filter, versionSet]);

  const sidebarItemCount = sidebarGroups.reduce((n, g) => n + g.items.length, 0);

  const displayItems = useMemo(() => {
    if (selectedSubfolder === "All") return filtered;

    if (filter === "Model") {
      if (selectedSubfolder === "Local") {
        return filtered.filter((item) => isExtraNetwork(item) && !isReferenceName(item.name) && !item.name.startsWith("Diffusers/"));
      }
      if (selectedSubfolder === "Diffusers") {
        return filtered.filter((item) => isExtraNetwork(item) && item.name.startsWith("Diffusers/"));
      }
      if (selectedSubfolder === "Reference") {
        return filtered.filter((item) => isExtraNetwork(item) && isReferenceName(item.name) && item.tags.length === 0);
      }
      const tagCat = TAG_CATEGORIES.find((c) => c === selectedSubfolder);
      if (tagCat) {
        return filtered.filter((item) => isExtraNetwork(item) && itemHasTag(item, tagCat.toLowerCase()));
      }
      if (versionSet.has(selectedSubfolder)) {
        return filtered.filter((item) => isExtraNetwork(item) && item.version === selectedSubfolder);
      }
      // Real subdir
      const prefix = selectedSubfolder + "/";
      const altPrefix = "models/" + prefix;
      return filtered.filter((item) => item.name.startsWith(prefix) || item.name.startsWith(altPrefix));
    }

    if (filter === "Style") {
      if (selectedSubfolder === "Local") return filtered.filter((item) => !isReferenceName(item.name));
      if (selectedSubfolder === "Reference") return filtered.filter((item) => isReferenceName(item.name));
    }

    // LoRA, Wildcards, Embedding, VAE — version or path-based filter
    if (versionSet.has(selectedSubfolder)) {
      return filtered.filter((item) => isExtraNetwork(item) && item.version === selectedSubfolder);
    }
    const prefix = selectedSubfolder + "/";
    return filtered.filter((item) => item.name.startsWith(prefix));
  }, [filtered, selectedSubfolder, filter, versionSet]);

  function handleFilterChange(t: TypeFilter) {
    setFilter(t);
    setSelectedSubfolder("All");
  }

  function handleClick(item: ExtraNetworkV2 | PromptStyle) {
    if ("type" in item && item.type) {
      const network = item as ExtraNetworkV2;
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
      } else if (t === "wildcards") {
        const current = useGenerationStore.getState().prompt;
        const tag = `__${network.name}__`;
        useGenerationStore.getState().setParam("prompt", current ? `${current} ${tag}` : tag);
      } else if (t === "vae") {
        setOptions.mutate({ sd_vae: network.title ?? network.name });
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
    <div>
      {/* Header */}
      <div className="sticky top-0 z-10 bg-card p-2 space-y-2 border-b border-border">
        <div className="flex items-center gap-1.5 flex-wrap">
          {TYPE_FILTERS.map((t) => (
            <button
              key={t}
              type="button"
              onClick={() => handleFilterChange(t)}
              className={cn(
                "px-2 py-0.5 rounded-full text-2xs font-medium transition-colors",
                filter === t ? "bg-accent text-accent-foreground" : "bg-muted text-muted-foreground hover:text-foreground",
              )}
            >
              {t}
            </button>
          ))}
          <button
            type="button"
            disabled={refreshNetworks.isPending}
            onClick={() => refreshNetworks.mutate()}
            className="ml-auto p-1 text-muted-foreground hover:text-foreground disabled:opacity-50"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${refreshNetworks.isPending ? "animate-spin" : ""}`} />
          </button>
        </div>
        <Input
          placeholder="Search..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="h-6 text-2xs"
        />
      </div>

      {/* Sidebar + Grid */}
      <div className="flex">
        {sidebarItemCount > 2 && (
          <div className="sticky top-[5.5rem] self-start w-28 shrink-0 border-r border-border max-h-[calc(100dvh-10rem)] overflow-y-auto">
            {sidebarGroups.map((group, gi) => (
              <div key={group.header ?? gi}>
                {group.header && (
                  <div className="px-2 pt-2 pb-0.5 text-4xs font-semibold uppercase tracking-wider text-muted-foreground/60 border-t border-border">
                    {group.header}
                  </div>
                )}
                {group.items.map((dir) => (
                  <button
                    key={dir}
                    type="button"
                    onClick={() => setSelectedSubfolder(dir)}
                    className={cn(
                      "w-full text-left px-2 py-1 text-2xs truncate transition-colors",
                      selectedSubfolder === dir ? "bg-accent text-accent-foreground font-medium" : "text-muted-foreground hover:text-foreground hover:bg-muted/50",
                    )}
                  >
                    {dir}
                  </button>
                ))}
              </div>
            ))}
          </div>
        )}
        <div className="flex-1 min-w-0 p-2">
          {isLoading && <p className="text-xs text-muted-foreground p-2">Loading networks...</p>}
          <div className="grid grid-cols-[repeat(auto-fill,minmax(140px,1fr))] gap-1.5">
            {displayItems.map((item) => (
              <NetworkCard key={item.name} item={item} active={isItemActive(item, prompt, options)} onClick={() => handleClick(item)} />
            ))}
          </div>
          {!isLoading && displayItems.length === 0 && (
            <p className="text-xs text-muted-foreground text-center py-6">No results found.</p>
          )}
        </div>
      </div>
    </div>
  );
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function DetailRow({ label, value }: { label: string; value: string | null | undefined }) {
  if (!value) return null;
  return (
    <div className="flex gap-2 text-2xs leading-tight">
      <span className="text-muted-foreground shrink-0 w-16">{label}</span>
      <span className="truncate font-medium">{value}</span>
    </div>
  );
}

function NetworkDetailPopover({ item }: { item: ExtraNetworkV2 | PromptStyle }) {
  const [open, setOpen] = useState(false);
  const isNetwork = "type" in item && item.type;
  const network = isNetwork ? (item as ExtraNetworkV2) : null;
  const { data: detail, isLoading } = useNetworkDetail(network?.type ?? "", item.name, open && !!network);

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <button
          type="button"
          onClick={(e) => { e.stopPropagation(); }}
          className="p-0.5 rounded-full text-muted-foreground hover:text-foreground hover:bg-muted/80 transition-colors"
        >
          <Info className="h-3 w-3" />
        </button>
      </PopoverTrigger>
      <PopoverContent side="right" align="start" className="w-64 p-3 space-y-2" onClick={(e) => e.stopPropagation()}>
        <p className="text-xs font-semibold truncate">{item.name}</p>
        {!network ? (
          <StyleDetail item={item as PromptStyle} />
        ) : isLoading ? (
          <div className="flex items-center gap-1.5 text-xs text-muted-foreground py-2">
            <Loader2 className="h-3 w-3 animate-spin" />
            Loading...
          </div>
        ) : detail && detail.name ? (
          <div className="space-y-1">
            <DetailRow label="Type" value={detail.type} />
            <DetailRow label="Alias" value={detail.alias} />
            <DetailRow label="Hash" value={detail.hash} />
            <DetailRow label="Version" value={detail.version} />
            <DetailRow label="Size" value={detail.size != null ? formatBytes(detail.size) : null} />
            <DetailRow label="Modified" value={detail.mtime ? new Date(detail.mtime).toLocaleDateString() : null} />
            <DetailRow label="Tags" value={detail.tags?.replaceAll("|", ", ")} />
            <DetailRow label="File" value={detail.filename?.split("/").pop()} />
            {detail.description && (
              <p className="text-2xs text-muted-foreground pt-1 border-t border-border mt-1">{detail.description}</p>
            )}
          </div>
        ) : (
          <p className="text-2xs text-muted-foreground">No detail available.</p>
        )}
      </PopoverContent>
    </Popover>
  );
}

function StyleDetail({ item }: { item: PromptStyle }) {
  return (
    <div className="space-y-1">
      {item.prompt && (
        <div className="text-2xs">
          <span className="text-muted-foreground">Prompt: </span>
          <span className="break-words">{item.prompt}</span>
        </div>
      )}
      {item.negative_prompt && (
        <div className="text-2xs">
          <span className="text-muted-foreground">Negative: </span>
          <span className="break-words">{item.negative_prompt}</span>
        </div>
      )}
      {item.filename && <DetailRow label="File" value={item.filename.split("/").pop()} />}
    </div>
  );
}

function NetworkCard({ item, active, onClick }: { item: ExtraNetworkV2 | PromptStyle; active: boolean; onClick: () => void }) {
  const isNetwork = "type" in item && item.type;
  const network = isNetwork ? (item as ExtraNetworkV2) : null;
  const typeBadge = network ? (network.version || network.type) : "Style";
  const preview = item.preview;
  const previewUrl = preview
    ? preview.startsWith("data:") || preview.startsWith("http") ? preview : `${api.getBaseUrl()}${preview}`
    : null;

  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "flex flex-col rounded-md border overflow-hidden transition-colors text-left",
        active
          ? "border-primary bg-primary/15"
          : "border-border hover:border-primary/50 hover:bg-primary/5",
      )}
    >
      <div className="aspect-square w-full bg-muted/30 flex items-center justify-center overflow-hidden">
        {previewUrl ? (
          <img src={previewUrl} alt={item.name} className="w-full h-full object-cover" loading="lazy" />
        ) : (
          <ImageOff className="h-6 w-6 text-muted-foreground/40" />
        )}
      </div>
      <div className="p-1.5 space-y-0.5">
        <p className="text-2xs font-medium truncate leading-tight">{item.name}</p>
        <div className="flex items-center justify-between gap-1">
          <Badge variant="secondary" className="text-4xs px-1 py-0">{typeBadge}</Badge>
          <NetworkDetailPopover item={item} />
        </div>
      </div>
    </button>
  );
}
