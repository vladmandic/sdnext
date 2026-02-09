import { Folder, FolderOpen, ChevronRight, ChevronDown, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";

interface FolderCardProps {
  label: string;
  path: string;
  active: boolean;
  indent?: number;
  expanded?: boolean;
  hasChildren?: boolean;
  loading?: boolean;
  onSelect: () => void;
  onToggle?: () => void;
}

export function FolderCard({ label, path, active, indent = 0, expanded, hasChildren, loading, onSelect, onToggle }: FolderCardProps) {
  const Icon = active ? FolderOpen : Folder;
  const displayLabel = label || path.split("/").filter(Boolean).pop() || path;

  return (
    <div
      className={cn(
        "flex items-center w-full rounded-md text-xs transition-colors group",
        "hover:bg-accent/50",
        active ? "bg-accent text-accent-foreground" : "text-muted-foreground",
      )}
      style={{ paddingLeft: indent * 16 + 4 }}
    >
      {/* Expand/collapse toggle */}
      {loading ? (
        <span className="w-5 h-5 flex items-center justify-center flex-shrink-0">
          <Loader2 size={12} className="animate-spin text-muted-foreground" />
        </span>
      ) : hasChildren ? (
        <button
          onClick={(e) => { e.stopPropagation(); onToggle?.(); }}
          className="w-5 h-5 flex items-center justify-center flex-shrink-0 hover:text-foreground"
        >
          {expanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        </button>
      ) : (
        <span className="w-5 flex-shrink-0" />
      )}

      {/* Folder button */}
      <button
        onClick={onSelect}
        className="flex items-center gap-1.5 flex-1 min-w-0 py-1.5 pr-2"
        title={path}
      >
        <Icon size={13} className={cn("flex-shrink-0", active ? "text-primary" : "text-muted-foreground")} />
        <span className="truncate">{displayLabel}</span>
      </button>
    </div>
  );
}
