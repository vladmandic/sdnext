import { useState, useMemo } from "react";
import { Check, ChevronDown, ChevronRight, Download, Loader2, Bookmark, Ban, MessageSquareText, X, Heart, Star, ArrowDownToLine } from "lucide-react";
import { useCivitModel, useCivitDownload, useCivitResolvePath, useCivitBookmarks, useCivitAddBookmark, useCivitRemoveBookmark, useCivitBanned, useCivitAddBanned, useCivitRemoveBanned, useCivitCheckLocal } from "@/api/hooks/useCivitai";
import type { CivitVersion, CivitFile, CivitImage, CivitStats } from "@/api/types/civitai";
import { Dialog, DialogContent, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";

interface CivitModelDetailProps {
  modelId: number | null;
  onClose: () => void;
}

function formatSize(sizeKB: number): string {
  if (sizeKB >= 1_048_576) return `${(sizeKB / 1_048_576).toFixed(1)} GB`;
  if (sizeKB >= 1024) return `${(sizeKB / 1024).toFixed(1)} MB`;
  return `${sizeKB.toFixed(0)} KB`;
}

function formatCount(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return String(n);
}

function civitThumbnail(url: string, width = 450): string {
  return url.replace(/\/(width=\d+|original=true)\//, `/width=${width}/`);
}

function ImageStrip({ images, className }: { images: CivitImage[]; className?: string }) {
  const filtered = images.filter((img) => !img.url.toLowerCase().endsWith(".mp4"));
  if (filtered.length === 0) return null;
  return (
    <div className={`flex gap-2 overflow-x-auto pb-1 ${className ?? ""}`}>
      {filtered.slice(0, 8).map((img) => (
        <img key={img.id} src={civitThumbnail(img.url)} alt="" className="h-28 w-auto rounded object-cover shrink-0" loading="lazy" />
      ))}
    </div>
  );
}

function StatsRow({ stats, className }: { stats: CivitStats; className?: string }) {
  return (
    <div className={`flex items-center gap-3 text-xs text-muted-foreground ${className ?? ""}`}>
      <span className="flex items-center gap-1"><ArrowDownToLine className="h-3 w-3" />{formatCount(stats.downloadCount)}</span>
      <span className="flex items-center gap-1"><Heart className="h-3 w-3" />{formatCount(stats.favoriteCount)}</span>
      {stats.rating > 0 && <span className="flex items-center gap-1"><Star className="h-3 w-3" />{stats.rating.toFixed(1)}</span>}
    </div>
  );
}

function HtmlDescription({ html }: { html: string }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <div className="relative">
      <div
        className={`text-xs text-muted-foreground prose prose-invert prose-xs max-w-none overflow-hidden ${expanded ? "" : "max-h-20"}`}
        dangerouslySetInnerHTML={{ __html: html }}
      />
      {!expanded && (
        <button type="button" onClick={() => setExpanded(true)} className="text-xs text-primary hover:underline mt-1">
          Show more
        </button>
      )}
    </div>
  );
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
  const [showTriggers, setShowTriggers] = useState(false);
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
    <div className="border border-border/50 rounded-lg overflow-hidden">
      {/* Version header — always visible */}
      <button type="button" onClick={() => setOpen(!open)} className="flex items-center gap-3 w-full px-4 py-3 text-left hover:bg-muted/30 transition-colors">
        {open ? <ChevronDown className="h-4 w-4 shrink-0 text-muted-foreground" /> : <ChevronRight className="h-4 w-4 shrink-0 text-muted-foreground" />}
        <span className="text-sm font-medium truncate min-w-0">{version.name}</span>
        <Badge variant="outline" className="text-3xs px-1.5 py-0.5 shrink-0">{version.baseModel}</Badge>
        <StatsRow stats={version.stats} className="ml-auto shrink-0 !text-3xs !gap-2" />
      </button>

      {open && (
        <div className="px-4 pb-4 space-y-4">
          {/* Version images */}
          <ImageStrip images={version.images} className="!h-24 [&_img]:h-24" />

          {/* Version description */}
          {version.description && <HtmlDescription html={version.description} />}

          {/* Trigger words — collapsible */}
          {version.trainedWords.length > 0 && (
            <div>
              <button type="button" onClick={() => setShowTriggers(!showTriggers)} className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors">
                <MessageSquareText className="h-3.5 w-3.5 shrink-0" />
                <span>Trigger words ({version.trainedWords.length})</span>
                {showTriggers ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
              </button>
              {showTriggers && (
                <div className="mt-2 max-h-24 overflow-y-auto rounded-md bg-muted/30 p-3">
                  <div className="flex flex-wrap gap-1.5">
                    {version.trainedWords.map((w) => (
                      <Badge key={w} variant="secondary" className="text-3xs px-1.5 py-0.5 max-w-full">
                        <span className="truncate">{w}</span>
                      </Badge>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Files — grid layout with fixed columns so buttons can never be displaced */}
          {version.files.length > 0 && (
            <div className="rounded-md border border-border/30 overflow-hidden">
              {version.files.map((f, i) => {
                const localMatch = f.hashes.SHA256 ? localFiles[f.hashes.SHA256] : undefined;
                return (
                  <div key={f.id} className={`grid grid-cols-[1fr_5rem_2.5rem] items-center gap-2 px-3 py-2 text-xs ${i > 0 ? "border-t border-border/20" : ""}`}>
                    <span className="truncate min-w-0" title={f.name}>{f.name}</span>
                    <span className="text-muted-foreground text-right">{formatSize(f.sizeKB)}</span>
                    <span className="flex justify-center">
                      {localMatch ? (
                        <span title={`Downloaded: ${localMatch.filename}`}><Check className="h-4 w-4 text-green-500" /></span>
                      ) : (
                        <Button size="icon" variant="ghost" className="h-7 w-7" onClick={() => handleDownload(f)} disabled={download.isPending}>
                          {download.isPending ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Download className="h-3.5 w-3.5" />}
                        </Button>
                      )}
                    </span>
                  </div>
                );
              })}
            </div>
          )}

          {/* Download path */}
          {resolved?.path && <p className="text-xs text-muted-foreground truncate" title={resolved.path}>&rarr; {resolved.path}</p>}
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

  // Collect all images from all versions for the hero strip
  const allImages = useMemo(() => {
    if (!model) return [];
    const imgs: CivitImage[] = [];
    for (const v of model.modelVersions) {
      for (const img of v.images) {
        imgs.push(img);
      }
    }
    return imgs;
  }, [model]);

  return (
    <Dialog open={modelId !== null} onOpenChange={(open) => { if (!open) onClose(); }}>
      <DialogContent showCloseButton={false} className="sm:max-w-5xl max-h-[85vh] flex flex-col p-0 gap-0 overflow-hidden">
        {isLoading ? (
          <div className="flex items-center justify-center py-16">
            <Loader2 className="h-6 w-6 animate-spin" />
          </div>
        ) : model ? (
          <>
            {/* ── FIXED TOP BAR ── */}
            <div className="shrink-0 flex items-center gap-3 px-5 py-3 border-b border-border/50 bg-background">
              <DialogTitle className="text-base font-semibold line-clamp-2 min-w-0 flex-1">{model.name}</DialogTitle>
              <Badge variant="outline" className="text-3xs px-1.5 py-0.5 shrink-0">{model.type}</Badge>
              <div className="flex items-center gap-1 shrink-0 ml-2">
                <Button
                  size="icon"
                  variant="ghost"
                  className="h-7 w-7"
                  onClick={() => isBookmarked ? removeBookmark.mutate(model.name) : addBookmark.mutate(model.name)}
                  title={isBookmarked ? "Remove bookmark" : "Bookmark"}
                >
                  <Bookmark className={`h-4 w-4 ${isBookmarked ? "fill-primary text-primary" : "text-muted-foreground"}`} />
                </Button>
                <Button
                  size="icon"
                  variant="ghost"
                  className="h-7 w-7"
                  onClick={() => isBanned ? removeBan.mutate(model.name) : addBan.mutate(model.name)}
                  title={isBanned ? "Remove from banned" : "Ban this model"}
                >
                  <Ban className={`h-4 w-4 ${isBanned ? "fill-orange-500 text-orange-500" : "text-muted-foreground"}`} />
                </Button>
                <Button size="icon" variant="ghost" className="h-7 w-7" onClick={onClose} title="Close">
                  <X className="h-4 w-4" />
                </Button>
              </div>
              <DialogDescription className="sr-only">Model details and download options</DialogDescription>
            </div>

            {/* ── SCROLLABLE BODY ── */}
            <div className="flex-1 overflow-y-auto min-h-0">
              <div className="p-5 space-y-5">
                {/* Creator + stats */}
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-sm text-muted-foreground">by {model.creator.username}</span>
                  <span className="text-muted-foreground/40">·</span>
                  <StatsRow stats={model.stats} />
                </div>

                {/* Tags */}
                {model.tags.length > 0 && (
                  <div className="flex flex-wrap gap-1.5">
                    {model.tags.slice(0, 16).map((t) => (
                      <Badge key={t} variant="secondary" className="text-3xs px-1.5 py-0.5">{t}</Badge>
                    ))}
                  </div>
                )}

                {/* Image gallery */}
                <ImageStrip images={allImages} />

                {/* Model description */}
                {model.description && <HtmlDescription html={model.description} />}

                <Separator />

                {/* Versions */}
                <div className="space-y-3">
                  {model.modelVersions.map((v) => (
                    <VersionSection key={v.id} version={v} modelType={model.type} modelName={model.name} creatorName={model.creator.username} modelId={model.id} modelNsfw={model.nsfw} localFiles={localFiles} />
                  ))}
                </div>
              </div>
            </div>
          </>
        ) : (
          <p className="text-sm text-muted-foreground py-8 text-center">Model not found</p>
        )}
      </DialogContent>
    </Dialog>
  );
}
