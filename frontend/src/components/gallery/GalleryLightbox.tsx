import type { ResHistory } from "@/api/types/progress";
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { formatDuration } from "@/lib/utils";

interface GalleryLightboxProps {
  item: ResHistory | null;
  imageUrl: string | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function GalleryLightbox({ item, imageUrl, open, onOpenChange }: GalleryLightboxProps) {
  if (!item || !imageUrl) return null;

  const timestamp = item.timestamp ? new Date(item.timestamp * 1000) : null;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-[90vw] max-h-[90vh] p-0 gap-0 overflow-hidden">
        <DialogTitle className="sr-only">Image details</DialogTitle>
        <div className="flex h-[85vh]">
          {/* Image */}
          <div className="flex-1 flex items-center justify-center bg-black/50 min-w-0">
            <img
              src={imageUrl}
              alt={item.op}
              className="max-w-full max-h-full object-contain"
            />
          </div>

          {/* Metadata sidebar */}
          <div className="w-64 border-l border-border flex-shrink-0">
            <ScrollArea className="h-full">
              <div className="p-4 space-y-4">
                <div>
                  <h3 className="text-xs font-semibold text-foreground mb-2">Details</h3>
                  <div className="space-y-2">
                    <MetaRow label="Operation" value={item.op} />
                    <MetaRow label="Job" value={item.job} />
                    {timestamp && (
                      <MetaRow label="Time" value={timestamp.toLocaleString()} />
                    )}
                    {item.duration != null && (
                      <MetaRow label="Duration" value={formatDuration(item.duration)} />
                    )}
                    {item.outputs.length > 1 && (
                      <MetaRow label="Images" value={String(item.outputs.length)} />
                    )}
                  </div>
                </div>

                <Separator />

                <div>
                  <h3 className="text-xs font-semibold text-foreground mb-2">Files</h3>
                  <div className="space-y-1">
                    {item.outputs.map((output, idx) => {
                      const filename = output.split("/").pop() ?? output;
                      return (
                        <Badge key={idx} variant="secondary" className="text-[10px] font-mono break-all block w-fit max-w-full">
                          {filename}
                        </Badge>
                      );
                    })}
                  </div>
                </div>
              </div>
            </ScrollArea>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

function MetaRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-start gap-2">
      <span className="text-[10px] text-muted-foreground w-16 flex-shrink-0">{label}</span>
      <span className="text-[11px] text-foreground break-all">{value}</span>
    </div>
  );
}
