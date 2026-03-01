import { useState } from "react";
import { useBrowserFolders } from "@/api/hooks/useGallery";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { FolderInput, FolderOpen, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";

interface MoveToDialogProps {
  open: boolean;
  count: number;
  isPending: boolean;
  onConfirm: (destination: string) => void;
  onCancel: () => void;
}

export function MoveToDialog({ open, count, isPending, onConfirm, onCancel }: MoveToDialogProps) {
  const { data: folders, isLoading } = useBrowserFolders();
  const [selected, setSelected] = useState<string | null>(null);

  const handleConfirm = () => {
    if (selected) onConfirm(selected);
  };

  return (
    <Dialog open={open} onOpenChange={(v) => !v && onCancel()}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <FolderInput size={18} />
            Move {count} {count === 1 ? "file" : "files"}
          </DialogTitle>
          <DialogDescription>
            Select a destination folder.
          </DialogDescription>
        </DialogHeader>
        <ScrollArea className="max-h-64 border rounded-md">
          <div className="p-1.5 space-y-0.5">
            {isLoading && (
              <div className="flex items-center justify-center py-4 text-muted-foreground gap-2">
                <Loader2 size={14} className="animate-spin" />
                <span className="text-xs">Loading folders...</span>
              </div>
            )}
            {folders?.map((f) => (
              <button
                key={f.path}
                onClick={() => setSelected(f.path)}
                className={cn(
                  "w-full flex items-center gap-2 px-2 py-1.5 rounded-sm text-left text-sm transition-colors",
                  selected === f.path
                    ? "bg-primary/10 text-primary border border-primary/30"
                    : "hover:bg-accent text-foreground border border-transparent",
                )}
              >
                <FolderOpen size={14} className="flex-shrink-0 text-muted-foreground" />
                <span className="truncate">{f.label}</span>
              </button>
            ))}
          </div>
        </ScrollArea>
        <DialogFooter>
          <Button variant="outline" onClick={onCancel} disabled={isPending}>Cancel</Button>
          <Button onClick={handleConfirm} disabled={isPending || !selected}>
            {isPending ? "Moving..." : "Move"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
