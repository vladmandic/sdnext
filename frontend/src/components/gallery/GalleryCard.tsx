import type { ResHistory } from "@/api/types/progress";
import { cn } from "@/lib/utils";

interface GalleryCardProps {
  item: ResHistory;
  imageUrl: string;
  onClick: () => void;
}

export function GalleryCard({ item, imageUrl, onClick }: GalleryCardProps) {
  const timestamp = item.timestamp ? new Date(item.timestamp * 1000) : null;

  return (
    <button
      onClick={onClick}
      className={cn(
        "group relative aspect-square rounded-lg overflow-hidden bg-muted border border-border",
        "hover:border-primary/50 hover:ring-1 hover:ring-primary/20 transition-all cursor-pointer",
      )}
    >
      <img
        src={imageUrl}
        alt={item.op}
        loading="lazy"
        className="w-full h-full object-cover"
      />
      {/* Hover overlay */}
      <div className="absolute inset-0 bg-gradient-to-t from-black/70 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity">
        <div className="absolute bottom-0 left-0 right-0 p-2">
          <p className="text-[10px] text-white/80 font-medium truncate">{item.op}</p>
          {timestamp && (
            <p className="text-[9px] text-white/60">{formatRelativeTime(timestamp)}</p>
          )}
        </div>
      </div>
    </button>
  );
}

function formatRelativeTime(date: Date): string {
  const now = Date.now();
  const diff = now - date.getTime();
  const minutes = Math.floor(diff / 60000);
  if (minutes < 1) return "Just now";
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}
