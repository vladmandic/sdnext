import { useState, useRef } from "react";
import { ChevronDown, ChevronRight, History, Loader2, Search, X } from "lucide-react";
import { useCivitTags, useCivitHistory, useCivitClearHistory } from "@/api/hooks/useCivitai";
import { useDebounce } from "@/hooks/useDebounce";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

interface CivitSearchBarProps {
  query: string;
  tag: string;
  onQueryChange: (v: string) => void;
  onTagChange: (v: string) => void;
  onSearch: () => void;
  onHistorySelect: (query: string, tag: string) => void;
  isLoading: boolean;
}

export function CivitSearchBar({ query, tag, onQueryChange, onTagChange, onSearch, onHistorySelect, isLoading }: CivitSearchBarProps) {
  const [tagFocused, setTagFocused] = useState(false);
  const [historyOpen, setHistoryOpen] = useState(false);
  const blurTimeout = useRef<ReturnType<typeof setTimeout>>(null);
  const debouncedTag = useDebounce(tag, 300);
  const { data: tagResults } = useCivitTags(debouncedTag, debouncedTag.length >= 2);
  const showSuggestions = tagFocused && tag.length >= 2 && (tagResults?.items?.length ?? 0) > 0;

  const { data: historyData } = useCivitHistory();
  const clearHistory = useCivitClearHistory();
  const entries = historyData?.history ?? [];

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
    onQueryChange("");
    onTagChange(name);
    setTagFocused(false);
  }

  return (
    <div className="space-y-1">
      <div className="flex gap-2">
        <Input
          placeholder="Search CivitAI..."
          value={query}
          onChange={(e) => { onQueryChange(e.target.value); if (e.target.value) onTagChange(""); }}
          onKeyDown={handleKeyDown}
          disabled={!!tag}
          className="h-6 text-2xs flex-1"
        />
        <div className="relative w-36">
          <Input
            placeholder="Tag..."
            value={tag}
            onChange={(e) => { onTagChange(e.target.value); if (e.target.value) onQueryChange(""); }}
            onKeyDown={handleKeyDown}
            onFocus={handleTagFocus}
            onBlur={handleTagBlur}
            disabled={!!query}
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
        <Button size="sm" variant="secondary" onClick={onSearch} disabled={isLoading} className="shrink-0">
          {isLoading ? <Loader2 className="h-3 w-3 animate-spin" /> : <Search className="h-3 w-3" />}
          Search
        </Button>
      </div>
      {entries.length > 0 && (
        <div>
          <button type="button" onClick={() => setHistoryOpen(!historyOpen)} className="flex items-center gap-1 text-3xs text-muted-foreground hover:text-foreground transition-colors py-0.5">
            <History className="h-2.5 w-2.5" />
            <span>Search history</span>
            {historyOpen ? <ChevronDown className="h-2.5 w-2.5" /> : <ChevronRight className="h-2.5 w-2.5" />}
          </button>
          {historyOpen && (
            <div className="flex items-center gap-1.5 flex-wrap pt-1">
              {entries.slice(0, 10).map((e) => (
                <Badge
                  key={`${e.type}-${e.term}`}
                  variant="secondary"
                  className="text-2xs cursor-pointer hover:bg-muted"
                  onClick={() => { onHistorySelect(e.type === "query" ? e.term : "", e.type === "tag" ? e.term : ""); }}
                >
                  {e.type === "tag" ? `#${e.term}` : e.term}
                </Badge>
              ))}
              <Button variant="ghost" size="icon" className="h-5 w-5" onClick={() => clearHistory.mutate()} title="Clear search history">
                <X className="h-3 w-3" />
              </Button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
