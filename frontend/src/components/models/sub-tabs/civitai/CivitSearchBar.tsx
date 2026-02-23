import { useState, useRef } from "react";
import { Loader2, Search } from "lucide-react";
import { useCivitTags } from "@/api/hooks/useCivitai";
import { useDebounce } from "@/hooks/useDebounce";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";

interface CivitSearchBarProps {
  query: string;
  tag: string;
  onQueryChange: (v: string) => void;
  onTagChange: (v: string) => void;
  onSearch: () => void;
  isLoading: boolean;
}

export function CivitSearchBar({ query, tag, onQueryChange, onTagChange, onSearch, isLoading }: CivitSearchBarProps) {
  const [tagFocused, setTagFocused] = useState(false);
  const blurTimeout = useRef<ReturnType<typeof setTimeout>>(null);
  const debouncedTag = useDebounce(tag, 300);
  const { data: tagResults } = useCivitTags(debouncedTag, debouncedTag.length >= 2);
  const showSuggestions = tagFocused && tag.length >= 2 && (tagResults?.items?.length ?? 0) > 0;

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter") onSearch();
  }

  function handleTagFocus() {
    if (blurTimeout.current) clearTimeout(blurTimeout.current);
    setTagFocused(true);
  }

  function handleTagBlur() {
    blurTimeout.current = setTimeout(() => setTagFocused(false), 150);
  }

  function selectTag(name: string) {
    onTagChange(name);
    setTagFocused(false);
  }

  return (
    <div className="flex gap-2">
      <Input
        placeholder="Search CivitAI..."
        value={query}
        onChange={(e) => onQueryChange(e.target.value)}
        onKeyDown={handleKeyDown}
        className="h-6 text-2xs flex-1"
      />
      <div className="relative w-36">
        <Input
          placeholder="Tag..."
          value={tag}
          onChange={(e) => onTagChange(e.target.value)}
          onKeyDown={handleKeyDown}
          onFocus={handleTagFocus}
          onBlur={handleTagBlur}
          className="h-6 text-2xs"
        />
        {showSuggestions && (
          <div className="absolute top-full left-0 right-0 z-50 mt-1 max-h-48 overflow-y-auto rounded-md border border-border bg-popover shadow-md">
            {tagResults!.items.map((t) => (
              <button
                key={t.name}
                type="button"
                className="flex w-full items-center justify-between px-2 py-1.5 text-xs hover:bg-muted/50"
                onMouseDown={(e) => e.preventDefault()}
                onClick={() => selectTag(t.name)}
              >
                <span className="truncate">{t.name}</span>
                <span className="ml-2 shrink-0 text-3xs text-muted-foreground">{t.modelCount.toLocaleString()}</span>
              </button>
            ))}
          </div>
        )}
      </div>
      <Button size="sm" variant="secondary" onClick={onSearch} disabled={isLoading || (!query && !tag)} className="shrink-0">
        {isLoading ? <Loader2 className="h-3 w-3 animate-spin" /> : <Search className="h-3 w-3" />}
        Search
      </Button>
    </div>
  );
}
