import { Loader2, Search } from "lucide-react";
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
  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter") onSearch();
  }

  return (
    <div className="flex gap-2">
      <Input
        placeholder="Search CivitAI..."
        value={query}
        onChange={(e) => onQueryChange(e.target.value)}
        onKeyDown={handleKeyDown}
        className="h-7 text-xs flex-1"
      />
      <Input
        placeholder="Tag..."
        value={tag}
        onChange={(e) => onTagChange(e.target.value)}
        onKeyDown={handleKeyDown}
        className="h-7 text-xs w-24"
      />
      <Button size="sm" variant="secondary" onClick={onSearch} disabled={isLoading || (!query && !tag)} className="shrink-0">
        {isLoading ? <Loader2 className="h-3 w-3 animate-spin" /> : <Search className="h-3 w-3" />}
        Search
      </Button>
    </div>
  );
}
