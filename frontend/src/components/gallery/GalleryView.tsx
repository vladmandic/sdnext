import { useState, useMemo } from "react";
import { useHistory } from "@/api/hooks/useGallery";
import type { ResHistory } from "@/api/types/progress";
import { GalleryGrid } from "./GalleryGrid";
import { GalleryLightbox } from "./GalleryLightbox";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Search, Loader2 } from "lucide-react";

export function GalleryView() {
  const { data: history, isLoading } = useHistory();

  const [filter, setFilter] = useState("");
  const [sortOrder, setSortOrder] = useState<"newest" | "oldest">("newest");
  const [lightboxItem, setLightboxItem] = useState<ResHistory | null>(null);
  const [lightboxImage, setLightboxImage] = useState<string | null>(null);
  const [lightboxOpen, setLightboxOpen] = useState(false);

  const filtered = useMemo(() => {
    if (!history) return [];
    let items = history.filter((h) => h.outputs.length > 0);
    if (filter) {
      const lower = filter.toLowerCase();
      items = items.filter(
        (h) =>
          h.op.toLowerCase().includes(lower) ||
          h.job.toLowerCase().includes(lower) ||
          h.outputs.some((o) => o.toLowerCase().includes(lower)),
      );
    }
    if (sortOrder === "oldest") {
      items = [...items].reverse();
    }
    return items;
  }, [history, filter, sortOrder]);

  function handleImageClick(item: ResHistory, imageUrl: string) {
    setLightboxItem(item);
    setLightboxImage(imageUrl);
    setLightboxOpen(true);
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full text-muted-foreground gap-2">
        <Loader2 size={16} className="animate-spin" />
        <span className="text-sm">Loading gallery...</span>
      </div>
    );
  }

  const totalImages = filtered.reduce((acc, h) => acc + h.outputs.length, 0);

  return (
    <div className="flex flex-col h-full">
      {/* Filter bar */}
      <div className="flex items-center gap-2 px-4 py-2 border-b border-border flex-shrink-0">
        <div className="relative flex-1 max-w-xs">
          <Search size={14} className="absolute left-2 top-1/2 -translate-y-1/2 text-muted-foreground" />
          <Input
            placeholder="Filter images..."
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            className="h-7 text-xs pl-7"
          />
        </div>
        <Select value={sortOrder} onValueChange={(v) => setSortOrder(v as "newest" | "oldest")}>
          <SelectTrigger size="sm" className="w-28 h-7 text-xs">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="newest" className="text-xs">Newest first</SelectItem>
            <SelectItem value="oldest" className="text-xs">Oldest first</SelectItem>
          </SelectContent>
        </Select>
        <span className="text-[10px] text-muted-foreground tabular-nums">
          {totalImages} image{totalImages !== 1 ? "s" : ""}
        </span>
      </div>

      {/* Grid */}
      <ScrollArea className="flex-1">
        <GalleryGrid items={filtered} onImageClick={handleImageClick} />
      </ScrollArea>

      {/* Lightbox */}
      <GalleryLightbox
        item={lightboxItem}
        imageUrl={lightboxImage}
        open={lightboxOpen}
        onOpenChange={setLightboxOpen}
      />
    </div>
  );
}
