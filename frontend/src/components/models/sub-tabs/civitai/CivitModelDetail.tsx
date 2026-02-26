import { useState, useMemo } from "react";
import { Check, ChevronDown, ChevronRight, Download, Loader2, Bookmark, Ban } from "lucide-react";
import { useCivitModel, useCivitDownload, useCivitResolvePath, useCivitBookmarks, useCivitAddBookmark, useCivitRemoveBookmark, useCivitBanned, useCivitAddBanned, useCivitRemoveBanned, useCivitCheckLocal } from "@/api/hooks/useCivitai";
import type { CivitVersion, CivitFile } from "@/api/types/civitai";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";

interface CivitModelDetailProps {
  modelId: number | null;
  onClose: () => void;
}

function formatSize(sizeKB: number): string {
  if (sizeKB >= 1_048_576) return `${(sizeKB / 1_048_576).toFixed(1)} GB`;
  if (sizeKB >= 1024) return `${(sizeKB / 1024).toFixed(1)} MB`;
  return `${sizeKB.toFixed(0)} KB`;
}

interface VersionSectionProps {
  version: CivitVersion;
  modelType: string;
  modelName: string;
  creatorName: string;
  modelId: number;
  modelNsfw: boolean;
  localFiles: Record<string, { filename: string; type: string }>;
}

function VersionSection({ version, modelType, modelName, creatorName, modelId, modelNsfw, localFiles }: VersionSectionProps) {
  const [open, setOpen] = useState(false);
  const download = useCivitDownload();

  const resolveParams = useMemo(() => ({
    model_type: modelType,
    model_name: modelName,
    base_model: version.baseModel,
    creator: creatorName,
    model_id: String(modelId),
    version_id: String(version.id),
    version_name: version.name,
    nsfw: String(modelNsfw),
  }), [modelType, modelName, version.baseModel, creatorName, modelId, version.id, version.name, modelNsfw]);

  const { data: resolved } = useCivitResolvePath(resolveParams, open);

  function handleDownload(file: CivitFile) {
    download.mutate({
      url: file.downloadUrl,
      filename: file.name,
      model_type: modelType,
      expected_hash: file.hashes.SHA256 ?? undefined,
      model_name: modelName,
      base_model: version.baseModel,
      creator: creatorName,
      model_id: modelId,
      version_id: version.id,
      version_name: version.name,
      nsfw: modelNsfw,
    });
  }

  return (
    <div className="border border-border/50 rounded-md">
      <button type="button" onClick={() => setOpen(!open)} className="flex items-center gap-2 w-full px-3 py-2 text-left hover:bg-muted/30">
        {open ? <ChevronDown className="h-3 w-3 shrink-0" /> : <ChevronRight className="h-3 w-3 shrink-0" />}
        <span className="text-xs font-medium truncate">{version.name}</span>
        <Badge variant="outline" className="text-4xs px-1 py-0 shrink-0">{version.baseModel}</Badge>
      </button>
      {open && (
        <div className="px-3 pb-3 space-y-2">
          {version.trainedWords.length > 0 && (
            <div className="flex flex-wrap gap-1">
              {version.trainedWords.map((w) => (
                <Badge key={w} variant="secondary" className="text-4xs px-1 py-0">{w}</Badge>
              ))}
            </div>
          )}
          {version.files.length > 0 && (
            <div className="space-y-1">
              {version.files.map((f) => {
                const localMatch = f.hashes.SHA256 ? localFiles[f.hashes.SHA256] : undefined;
                return (
                  <div key={f.id} className="flex items-center gap-2 text-2xs">
                    <span className="truncate flex-1" title={f.name}>{f.name}</span>
                    <span className="text-muted-foreground shrink-0">{formatSize(f.sizeKB)}</span>
                    {localMatch ? (
                      <span className="h-6 w-6 shrink-0 flex items-center justify-center" title={`Downloaded: ${localMatch.filename}`}>
                        <Check className="h-3 w-3 text-green-500" />
                      </span>
                    ) : (
                      <Button
                        size="icon"
                        variant="ghost"
                        className="h-6 w-6 shrink-0"
                        onClick={() => handleDownload(f)}
                        disabled={download.isPending}
                      >
                        {download.isPending ? <Loader2 className="h-3 w-3 animate-spin" /> : <Download className="h-3 w-3" />}
                      </Button>
                    )}
                  </div>
                );
              })}
            </div>
          )}
          {resolved?.path && <p className="text-3xs text-muted-foreground truncate" title={resolved.path}>&rarr; {resolved.path}</p>}
        </div>
      )}
    </div>
  );
}

export function CivitModelDetail({ modelId, onClose }: CivitModelDetailProps) {
  const { data: model, isLoading } = useCivitModel(modelId);
  const { data: bookmarks } = useCivitBookmarks();
  const { data: banned } = useCivitBanned();
  const addBookmark = useCivitAddBookmark();
  const removeBookmark = useCivitRemoveBookmark();
  const addBan = useCivitAddBanned();
  const removeBan = useCivitRemoveBanned();

  const allHashes = useMemo(() => {
    if (!model) return [];
    const hashes: string[] = [];
    for (const v of model.modelVersions) {
      for (const f of v.files) {
        if (f.hashes.SHA256) hashes.push(f.hashes.SHA256);
      }
    }
    return hashes;
  }, [model]);
  const { data: localCheck } = useCivitCheckLocal(allHashes);
  const localFiles = localCheck?.found ?? {};

  const isBookmarked = model ? (bookmarks?.some((b) => b.name === model.name) ?? false) : false;
  const isBanned = model ? (banned?.some((b) => b.name === model.name) ?? false) : false;

  return (
    <Dialog open={modelId !== null} onOpenChange={(open) => { if (!open) onClose(); }}>
      <DialogContent className="sm:max-w-lg max-h-[80vh] flex flex-col">
        {isLoading ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="h-5 w-5 animate-spin" />
          </div>
        ) : model ? (
          <>
            <DialogHeader>
              <DialogTitle className="flex items-center gap-2 text-sm">
                {model.name}
                <Badge variant="outline" className="text-4xs px-1 py-0">{model.type}</Badge>
                <button
                  type="button"
                  onClick={() => isBookmarked ? removeBookmark.mutate(model.name) : addBookmark.mutate(model.name)}
                  className="p-1 rounded hover:bg-muted/50 transition-colors"
                  title={isBookmarked ? "Remove bookmark" : "Bookmark"}
                >
                  <Bookmark className={`h-3.5 w-3.5 ${isBookmarked ? "fill-primary text-primary" : "text-muted-foreground"}`} />
                </button>
                <button
                  type="button"
                  onClick={() => isBanned ? removeBan.mutate(model.name) : addBan.mutate(model.name)}
                  className="p-1 rounded hover:bg-muted/50 transition-colors"
                  title={isBanned ? "Remove from banned" : "Ban this model"}
                >
                  <Ban className={`h-3.5 w-3.5 ${isBanned ? "fill-orange-500 text-orange-500" : "text-muted-foreground"}`} />
                </button>
              </DialogTitle>
              <DialogDescription className="text-2xs">
                by {model.creator.username}
              </DialogDescription>
            </DialogHeader>
            {model.tags.length > 0 && (
              <div className="flex flex-wrap gap-1">
                {model.tags.slice(0, 12).map((t) => (
                  <Badge key={t} variant="secondary" className="text-4xs px-1 py-0">{t}</Badge>
                ))}
              </div>
            )}
            <ScrollArea className="flex-1 min-h-0">
              <div className="space-y-2 pr-3">
                {model.modelVersions.map((v) => (
                  <VersionSection key={v.id} version={v} modelType={model.type} modelName={model.name} creatorName={model.creator.username} modelId={model.id} modelNsfw={model.nsfw} localFiles={localFiles} />
                ))}
              </div>
            </ScrollArea>
          </>
        ) : (
          <p className="text-xs text-muted-foreground py-4 text-center">Model not found</p>
        )}
      </DialogContent>
    </Dialog>
  );
}
