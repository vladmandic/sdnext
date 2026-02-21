import { useState, useRef } from "react";
import type { CivitOptions } from "@/api/types/civitai";
import { useCivitCreators } from "@/api/hooks/useCivitai";
import { useDebounce } from "@/hooks/useDebounce";
import { Combobox } from "@/components/ui/combobox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";

interface CivitFiltersProps {
  options: CivitOptions | undefined;
  type: string;
  sort: string;
  period: string;
  baseModel: string;
  creator: string;
  nsfw: boolean;
  favorites: boolean;
  tokenConfigured: boolean;
  onTypeChange: (v: string) => void;
  onSortChange: (v: string) => void;
  onPeriodChange: (v: string) => void;
  onBaseModelChange: (v: string) => void;
  onCreatorChange: (v: string) => void;
  onNsfwChange: (v: boolean) => void;
  onFavoritesChange: (v: boolean) => void;
}

export function CivitFilters({ options, type, sort, period, baseModel, creator, nsfw, favorites, tokenConfigured, onTypeChange, onSortChange, onPeriodChange, onBaseModelChange, onCreatorChange, onNsfwChange, onFavoritesChange }: CivitFiltersProps) {
  const types = ["", ...(options?.types ?? [])];
  const sorts = ["", ...(options?.sort ?? [])];
  const periods = ["", ...(options?.period ?? [])];
  const baseModels = ["", ...(options?.base_models ?? [])];

  return (
    <div className="grid grid-cols-2 gap-2">
        <div>
          <Label className="text-[11px]">Type</Label>
          <Combobox value={type} onValueChange={onTypeChange} options={types} placeholder="All types" className="h-7 text-xs" />
        </div>
        <div>
          <Label className="text-[11px]">Sort</Label>
          <Combobox value={sort} onValueChange={onSortChange} options={sorts} placeholder="Default" className="h-7 text-xs" />
        </div>
        <div>
          <Label className="text-[11px]">Period</Label>
          <Combobox value={period} onValueChange={onPeriodChange} options={periods} placeholder="All time" className="h-7 text-xs" />
        </div>
        <div>
          <Label className="text-[11px]">Base model</Label>
          <Combobox value={baseModel} onValueChange={onBaseModelChange} options={baseModels} placeholder="Any" className="h-7 text-xs" />
        </div>
        <CreatorInput value={creator} onChange={onCreatorChange} />
        <div className="flex flex-col justify-end gap-1.5 pb-0.5">
          <div className="flex items-center gap-1.5">
            <Switch id="civit-nsfw" size="sm" checked={nsfw} onCheckedChange={onNsfwChange} />
            <Label htmlFor="civit-nsfw" className="text-[11px]">NSFW</Label>
          </div>
          {tokenConfigured && (
            <div className="flex items-center gap-1.5">
              <Switch id="civit-favorites" size="sm" checked={favorites} onCheckedChange={onFavoritesChange} />
              <Label htmlFor="civit-favorites" className="text-[11px]">Favorites</Label>
            </div>
          )}
        </div>
    </div>
  );
}

function CreatorInput({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  const [focused, setFocused] = useState(false);
  const blurTimeout = useRef<ReturnType<typeof setTimeout>>(null);
  const debouncedValue = useDebounce(value, 300);
  const { data: results } = useCivitCreators(debouncedValue, debouncedValue.length >= 2);
  const showSuggestions = focused && value.length >= 2 && (results?.items?.length ?? 0) > 0;

  function handleFocus() {
    if (blurTimeout.current) clearTimeout(blurTimeout.current);
    setFocused(true);
  }

  function handleBlur() {
    blurTimeout.current = setTimeout(() => setFocused(false), 150);
  }

  function select(username: string) {
    onChange(username);
    setFocused(false);
  }

  return (
    <div className="relative">
      <Label className="text-[11px]">Creator</Label>
      <Input
        placeholder="Username..."
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onFocus={handleFocus}
        onBlur={handleBlur}
        className="h-7 text-xs"
      />
      {showSuggestions && (
        <div className="absolute top-full left-0 right-0 z-50 mt-1 max-h-48 overflow-y-auto rounded-md border border-border bg-popover shadow-md">
          {results!.items.map((c) => (
            <button
              key={c.username}
              type="button"
              className="flex w-full items-center justify-between px-2 py-1.5 text-xs hover:bg-muted/50"
              onMouseDown={(e) => e.preventDefault()}
              onClick={() => select(c.username)}
            >
              <span className="truncate">{c.username}</span>
              <span className="ml-2 shrink-0 text-[10px] text-muted-foreground">{c.modelCount.toLocaleString()}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
