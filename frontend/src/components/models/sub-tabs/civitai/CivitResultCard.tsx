import { Download, Heart, Ban } from "lucide-react";
import type { CivitModel } from "@/api/types/civitai";
import { useCivitFavorites, useCivitAddFavorite, useCivitRemoveFavorite, useCivitBanned, useCivitAddBanned, useCivitRemoveBanned } from "@/api/hooks/useCivitai";
import { Badge } from "@/components/ui/badge";

interface CivitResultCardProps {
  model: CivitModel;
  onClick: () => void;
}

function civitThumbnail(url: string, width = 80): string {
  // CivitAI CDN uses path-based sizing: /width=450/ or /original=true/
  return url.replace(/\/(width=\d+|original=true)\//, `/width=${width}/`);
}

function getPreviewUrl(model: CivitModel): string | null {
  for (const v of model.modelVersions) {
    for (const img of v.images) {
      if (img.url && !img.url.toLowerCase().endsWith(".mp4")) {
        return civitThumbnail(img.url);
      }
    }
  }
  return null;
}

function formatCount(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return String(n);
}

export function CivitResultCard({ model, onClick }: CivitResultCardProps) {
  const preview = getPreviewUrl(model);
  const { data: favorites } = useCivitFavorites();
  const { data: banned } = useCivitBanned();
  const addFav = useCivitAddFavorite();
  const removeFav = useCivitRemoveFavorite();
  const addBan = useCivitAddBanned();
  const removeBan = useCivitRemoveBanned();

  const isFavorited = favorites?.some((f) => f.name === model.name) ?? false;
  const isBanned = banned?.some((b) => b.name === model.name) ?? false;

  return (
    <button type="button" onClick={onClick} className="flex items-center gap-2.5 w-full px-2 py-1.5 hover:bg-muted/30 cursor-pointer text-left rounded-sm">
      <div className="w-10 h-10 rounded bg-muted/50 overflow-hidden shrink-0 flex items-center justify-center">
        {preview ? (
          <img src={preview} alt="" className="w-full h-full object-cover" loading="lazy" />
        ) : (
          <span className="text-[9px] text-muted-foreground">N/A</span>
        )}
      </div>
      <div className="flex-1 min-w-0">
        <div className="text-xs font-medium truncate">{model.name}</div>
        <div className="flex items-center gap-2 text-[10px] text-muted-foreground">
          <Badge variant="outline" className="text-[9px] px-1 py-0 shrink-0">{model.type}</Badge>
          {model.modelVersions[0]?.baseModel && <Badge variant="secondary" className="text-[9px] px-1 py-0 shrink-0">{model.modelVersions[0].baseModel}</Badge>}
          <span>{model.creator.username}</span>
          <span className="flex items-center gap-0.5">
            <Download className="h-2.5 w-2.5" />
            {formatCount(model.stats.downloadCount)}
          </span>
        </div>
      </div>
      <div className="flex items-center gap-0.5 shrink-0">
        <span
          role="button"
          tabIndex={0}
          onClick={(e) => { e.stopPropagation(); if (isFavorited) removeFav.mutate(model.name); else addFav.mutate(model.name); }}
          onKeyDown={(e) => { if (e.key === "Enter") { e.stopPropagation(); if (isFavorited) removeFav.mutate(model.name); else addFav.mutate(model.name); } }}
          className="p-1 rounded hover:bg-muted/50 transition-colors"
          title={isFavorited ? "Remove from favorites" : "Add to favorites"}
        >
          <Heart className={`h-3 w-3 ${isFavorited ? "fill-red-500 text-red-500" : "text-muted-foreground"}`} />
        </span>
        <span
          role="button"
          tabIndex={0}
          onClick={(e) => { e.stopPropagation(); if (isBanned) removeBan.mutate(model.name); else addBan.mutate(model.name); }}
          onKeyDown={(e) => { if (e.key === "Enter") { e.stopPropagation(); if (isBanned) removeBan.mutate(model.name); else addBan.mutate(model.name); } }}
          className="p-1 rounded hover:bg-muted/50 transition-colors"
          title={isBanned ? "Remove from banned" : "Ban this model"}
        >
          <Ban className={`h-3 w-3 ${isBanned ? "fill-orange-500 text-orange-500" : "text-muted-foreground"}`} />
        </span>
      </div>
    </button>
  );
}
