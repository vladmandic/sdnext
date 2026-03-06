import { useState } from "react";
import { Info, Loader2, ExternalLink, ImageOff, ChevronDown, ChevronRight } from "lucide-react";
import { toast } from "sonner";
import { useNetworkDetail } from "@/api/hooks/useNetworks";
import { useGenerationStore } from "@/stores/generationStore";
import type { ExtraNetworkV2, NetworkDetail, PromptStyleV2 } from "@/api/types/models";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Dialog, DialogContent, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { api } from "@/api/client";

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function DetailRow({ label, value }: { label: string; value: string | null | undefined }) {
  if (!value) return null;
  return (
    <div className="flex gap-2 text-xs leading-relaxed">
      <span className="text-muted-foreground shrink-0 w-18">{label}</span>
      <span className="font-medium break-words min-w-0">{value}</span>
    </div>
  );
}

function getCivitInfo(info: Record<string, unknown> | null | undefined) {
  if (!info || typeof info.id !== "number" || info.id <= 0) return null;
  const versions = Array.isArray(info.modelVersions) ? info.modelVersions as Array<Record<string, unknown>> : [];
  const firstVersion = versions[0];
  const trainedWords = Array.isArray(firstVersion?.trainedWords) ? (firstVersion.trainedWords as string[]).filter(Boolean) : [];
  const baseModel = typeof firstVersion?.baseModel === "string" ? firstVersion.baseModel : null;
  return {
    id: info.id as number,
    name: typeof info.name === "string" ? info.name : null,
    trainedWords,
    baseModel,
  };
}

function HtmlDescription({ html }: { html: string }) {
  const [expanded, setExpanded] = useState(false);
  const isHtml = /<[a-z][\s\S]*>/i.test(html);

  if (isHtml) {
    return (
      <div className="space-y-1.5">
        <div
          className={`prose prose-invert prose-sm max-w-none prose-p:my-1 prose-headings:my-2 prose-img:my-2 ${!expanded ? "max-h-28 overflow-hidden" : ""}`}
          dangerouslySetInnerHTML={{ __html: html }}
        />
        <button
          type="button"
          onClick={() => setExpanded(!expanded)}
          className="text-2xs text-primary hover:underline"
        >
          {expanded ? "Show less" : "Show more"}
        </button>
      </div>
    );
  }

  const lines = html.split("\n");
  const isLong = lines.length > 6 || html.length > 400;
  return (
    <div className="space-y-1.5">
      <p className={`text-xs text-muted-foreground whitespace-pre-wrap ${!expanded && isLong ? "max-h-28 overflow-hidden" : ""}`}>
        {html}
      </p>
      {isLong && (
        <button
          type="button"
          onClick={() => setExpanded(!expanded)}
          className="text-2xs text-primary hover:underline"
        >
          {expanded ? "Show less" : "Show more"}
        </button>
      )}
    </div>
  );
}

function TriggerWords({ words }: { words: string[] }) {
  const [expanded, setExpanded] = useState(false);

  function handleClick(word: string) {
    const current = useGenerationStore.getState().prompt;
    const updated = current ? `${current}, ${word}` : word;
    useGenerationStore.getState().setParam("prompt", updated);
    toast.success(`Added "${word}" to prompt`);
  }

  return (
    <div className="space-y-1.5">
      <button
        type="button"
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-1 text-xs font-medium hover:text-foreground transition-colors"
      >
        {expanded ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
        Trigger words ({words.length})
      </button>
      {expanded && (
        <div className="flex flex-wrap gap-1.5">
          {words.map((w, i) => (
            <button
              key={`${w}-${i}`}
              type="button"
              onClick={() => handleClick(w)}
              className="inline-flex items-center rounded-full border border-transparent bg-secondary text-secondary-foreground px-2 py-0.5 text-2xs font-medium cursor-pointer hover:bg-primary/20 select-none transition-colors"
            >
              {w}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

function NetworkDialogBody({ detail, previewUrl }: { detail: NetworkDetail; previewUrl: string | null }) {
  const civit = getCivitInfo(detail.info);
  const tags = detail.tags?.replaceAll("|", ", ").split(", ").filter(Boolean) ?? [];
  const hasPreview = !!previewUrl;

  return (
    <div className={`flex gap-5 min-h-0 ${hasPreview ? "" : "flex-col"}`}>
      {/* Preview image */}
      {hasPreview && (
        <div className="shrink-0 w-56 flex items-start justify-center">
          <img
            src={previewUrl}
            alt={detail.name}
            className="max-h-80 w-full object-contain rounded-md bg-muted/20"
          />
        </div>
      )}

      {/* Metadata + description */}
      <ScrollArea className="flex-1 min-w-0 min-h-0 max-h-[calc(80vh-5rem)]">
        <div className="space-y-3 pr-3">
          {/* Metadata rows */}
          <div className="space-y-1">
            <DetailRow label="Type" value={detail.type} />
            <DetailRow label="Alias" value={detail.alias} />
            <DetailRow label="Hash" value={detail.hash} />
            <DetailRow label="Version" value={detail.version} />
            <DetailRow label="Size" value={detail.size != null ? formatBytes(detail.size) : null} />
            <DetailRow label="Modified" value={detail.mtime ? new Date(detail.mtime).toLocaleDateString() : null} />
            <DetailRow label="File" value={detail.filename?.split("/").pop()} />
          </div>

          {/* Tags */}
          {tags.length > 0 && (
            <div className="space-y-1.5 pt-2 border-t border-border">
              <span className="text-xs font-medium">Tags</span>
              <div className="flex flex-wrap gap-1.5">
                {tags.map((t, i) => (
                  <button
                    key={`${t}-${i}`}
                    type="button"
                    onClick={() => {
                      const current = useGenerationStore.getState().prompt;
                      const tag = t.trim();
                      const updated = current ? `${current}, ${tag}` : tag;
                      useGenerationStore.getState().setParam("prompt", updated);
                      toast.success(`Added "${tag}" to prompt`);
                    }}
                    className="inline-flex items-center rounded-full border border-transparent bg-secondary text-secondary-foreground px-2 py-0.5 text-2xs font-medium cursor-pointer hover:bg-primary/20 select-none transition-colors"
                  >
                    {t.trim()}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* CivitAI trigger words */}
          {civit && civit.trainedWords.length > 0 && (
            <div className="pt-2 border-t border-border">
              <TriggerWords words={civit.trainedWords} />
            </div>
          )}

          {/* CivitAI link */}
          {civit && (
            <div className="pt-2 border-t border-border space-y-1">
              <div className="flex items-center gap-2">
                <span className="text-xs font-medium">CivitAI</span>
                <a
                  href={`https://civitai.com/models/${civit.id}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-muted-foreground hover:text-foreground transition-colors"
                >
                  <ExternalLink className="h-3 w-3" />
                </a>
              </div>
              {civit.name && <DetailRow label="Name" value={civit.name} />}
              {civit.baseModel && <DetailRow label="Base" value={civit.baseModel} />}
            </div>
          )}

          {/* Description */}
          {detail.description && (
            <div className="pt-2 border-t border-border space-y-1.5">
              <span className="text-xs font-medium">Description</span>
              <HtmlDescription html={detail.description} />
            </div>
          )}
        </div>
      </ScrollArea>
    </div>
  );
}

function StyleDialogBody({ item }: { item: PromptStyleV2 }) {
  const previewUrl = item.preview
    ? item.preview.startsWith("data:") || item.preview.startsWith("http") ? item.preview : `${api.getBaseUrl()}${item.preview}`
    : null;

  return (
    <div className={`flex gap-5 min-h-0 ${previewUrl ? "" : "flex-col"}`}>
      {previewUrl && (
        <div className="shrink-0 w-56 flex items-start justify-center">
          <img src={previewUrl} alt={item.name} className="max-h-80 w-full object-contain rounded-md bg-muted/20" />
        </div>
      )}
      <div className="flex-1 min-w-0 space-y-2">
        {item.prompt && (
          <div className="text-xs space-y-0.5">
            <span className="text-muted-foreground font-medium">Prompt</span>
            <p className="break-words bg-muted/30 rounded p-2 text-2xs">{item.prompt}</p>
          </div>
        )}
        {item.negative_prompt && (
          <div className="text-xs space-y-0.5">
            <span className="text-muted-foreground font-medium">Negative</span>
            <p className="break-words bg-muted/30 rounded p-2 text-2xs">{item.negative_prompt}</p>
          </div>
        )}
        {item.description && (
          <div className="pt-2 border-t border-border space-y-1.5">
            <span className="text-xs font-medium">Description</span>
            <HtmlDescription html={item.description} />
          </div>
        )}
        {item.filename && <DetailRow label="File" value={item.filename.split("/").pop()} />}
      </div>
    </div>
  );
}

export function NetworkDetailDialog({ item }: { item: ExtraNetworkV2 | PromptStyleV2 }) {
  const [open, setOpen] = useState(false);
  const isNetwork = "type" in item && item.type;
  const network = isNetwork ? (item as ExtraNetworkV2) : null;
  const { data: detail, isLoading } = useNetworkDetail(network?.type ?? "", item.name, open && !!network);

  const previewUrl = item.preview
    ? item.preview.startsWith("data:") || item.preview.startsWith("http") ? item.preview : `${api.getBaseUrl()}${item.preview}`
    : null;

  const typeBadge = network ? (network.version || network.type) : "Style";

  return (
    <>
      <button
        type="button"
        onClick={(e) => { e.stopPropagation(); setOpen(true); }}
        className="p-0.5 rounded-full text-muted-foreground hover:text-foreground hover:bg-muted/80 transition-colors"
      >
        <Info className="h-3 w-3" />
      </button>
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent
          showCloseButton
          className="sm:max-w-3xl max-h-[80vh] flex flex-col p-0 gap-0 overflow-hidden"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header — pr-10 reserves space for the absolute-positioned close button */}
          <div className="flex items-center gap-3 px-5 pr-10 py-3 border-b border-border/50 shrink-0">
            <DialogTitle className="text-sm font-semibold truncate flex-1 min-w-0">
              {item.name}
            </DialogTitle>
            <Badge variant="outline" className="text-2xs shrink-0">{typeBadge}</Badge>
            <DialogDescription className="sr-only">Details for {item.name}</DialogDescription>
          </div>

          {/* Body */}
          <div className="flex-1 min-h-0 overflow-y-auto p-5">
            {!network ? (
              <StyleDialogBody item={item as PromptStyleV2} />
            ) : isLoading ? (
              <div className="flex items-center gap-2 text-sm text-muted-foreground py-8 justify-center">
                <Loader2 className="h-4 w-4 animate-spin" />
                Loading details...
              </div>
            ) : detail && detail.name ? (
              <NetworkDialogBody detail={detail} previewUrl={previewUrl} />
            ) : (
              <div className="flex flex-col items-center gap-2 text-muted-foreground py-8">
                <ImageOff className="h-8 w-8 opacity-40" />
                <p className="text-sm">No details available</p>
              </div>
            )}
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}
