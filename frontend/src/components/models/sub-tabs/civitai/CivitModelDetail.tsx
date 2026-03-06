import { useState, useMemo, useCallback, useEffect } from "react";
import { createPortal } from "react-dom";
import { Check, ChevronDown, ChevronRight, ChevronLeft, Download, Loader2, Bookmark, Ban, MessageSquareText, X, Heart, Star, ArrowDownToLine, Image as ImageIcon, Send, Sparkles } from "lucide-react";
import { toast } from "sonner";
import { useCivitModel, useCivitDownload, useCivitResolvePath, useCivitBookmarks, useCivitAddBookmark, useCivitRemoveBookmark, useCivitBanned, useCivitAddBanned, useCivitRemoveBanned, useCivitCheckLocal, useCivitVersionImages } from "@/api/hooks/useCivitai";
import type { CivitVersion, CivitFile, CivitImage, CivitStats } from "@/api/types/civitai";
import { Dialog, DialogContent, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuSeparator, DropdownMenuLabel, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { fetchRemoteImage, sendImageToCanvas, sendFrameToVideoInit, sendPromptToGeneration, sendPromptToVideo, appendToGenerationPrompt } from "@/lib/sendTo";

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

function ImageStrip({ images, className, onImageClick }: { images: CivitImage[]; className?: string; onImageClick?: (index: number) => void }) {
  const filtered = images.filter((img) => !img.url.toLowerCase().endsWith(".mp4"));
  if (filtered.length === 0) return null;
  return (
    <div className={`flex gap-2 overflow-x-auto pb-1 ${className ?? ""}`}>
      {filtered.slice(0, 8).map((img, i) => (
        <img
          key={img.url}
          src={civitThumbnail(img.url)}
          alt=""
          className={`h-28 w-auto rounded object-cover shrink-0 ${onImageClick ? "cursor-pointer hover:ring-2 hover:ring-primary/60 transition-shadow" : ""}`}
          loading="lazy"
          onClick={onImageClick ? () => onImageClick(i) : undefined}
        />
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
  onImageClick?: (image: CivitImage) => void;
}

function VersionSection({ version, modelType, modelName, creatorName, modelId, modelNsfw, localFiles, onImageClick }: VersionSectionProps) {
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

  const downloadedCount = useMemo(() => {
    let count = 0;
    for (const f of version.files) {
      if (f.hashes.SHA256 && localFiles[f.hashes.SHA256]) count++;
    }
    return count;
  }, [version.files, localFiles]);
  const totalFiles = version.files.length;
  const hasAll = totalFiles > 0 && downloadedCount === totalFiles;
  const hasSome = downloadedCount > 0 && !hasAll;

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
        {hasAll && (
          <span title={`All ${totalFiles} file${totalFiles > 1 ? "s" : ""} downloaded`}>
            <Check className="h-4 w-4 shrink-0 text-green-500" />
          </span>
        )}
        {hasSome && (
          <span title={`${downloadedCount}/${totalFiles} file${totalFiles > 1 ? "s" : ""} downloaded`}>
            <Check className="h-4 w-4 shrink-0 text-yellow-500" />
          </span>
        )}
        <StatsRow stats={version.stats} className="ml-auto shrink-0 !text-3xs !gap-2" />
      </button>

      {open && (
        <div className="px-4 pb-4 space-y-4">
          {/* Version images — cap at 10 to match version API metadata coverage */}
          <ImageStrip images={version.images.filter((img) => !img.url.toLowerCase().endsWith(".mp4")).slice(0, 10)} className="!h-24 [&_img]:h-24" onImageClick={onImageClick ? (i) => {
            const filtered = version.images.filter((img) => !img.url.toLowerCase().endsWith(".mp4")).slice(0, 10);
            if (filtered[i]) onImageClick(filtered[i]);
          } : undefined} />

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
                    {version.trainedWords.map((w, i) => (
                      <Badge
                        key={`${w}-${i}`}
                        variant="secondary"
                        className="text-3xs px-1.5 py-0.5 max-w-full cursor-pointer hover:bg-accent"
                        onClick={() => { appendToGenerationPrompt(w); toast.success("Added to prompt", { description: w }); }}
                      >
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

// Metadata keys to display with human-readable labels
const META_DISPLAY_KEYS: [string, string][] = [
  ["prompt", "Prompt"],
  ["negativePrompt", "Negative"],
  ["steps", "Steps"],
  ["sampler", "Sampler"],
  ["cfgScale", "CFG"],
  ["seed", "Seed"],
  ["clipSkip", "CLIP Skip"],
  ["Size", "Size"],
];

function ImageLightbox({ images, index, onClose, onCloseAll, onNavigate }: { images: CivitImage[]; index: number; onClose: () => void; onCloseAll: () => void; onNavigate: (i: number) => void }) {
  const img = images[index];
  const meta = img?.meta ?? {};
  const hasPrompt = typeof meta.prompt === "string" && meta.prompt.length > 0;
  const hasNegative = typeof meta.negativePrompt === "string" && meta.negativePrompt.length > 0;

  const navigate = useCallback((delta: number) => {
    const next = index + delta;
    if (next >= 0 && next < images.length) onNavigate(next);
  }, [index, images.length, onNavigate]);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      switch (e.key) {
        case "Escape": e.stopImmediatePropagation(); onClose(); break;
        case "ArrowLeft": navigate(-1); break;
        case "ArrowRight": navigate(1); break;
      }
    };
    window.addEventListener("keydown", handler, true);
    return () => window.removeEventListener("keydown", handler, true);
  }, [onClose, navigate]);

  if (!img) return null;
  const fullUrl = img.url.replace(/\/(width=\d+)\//, "/original=true/");

  return (
    <div className="fixed inset-0 z-[100] bg-black/90 flex" onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      {/* Image area */}
      <div className="flex-1 flex items-center justify-center min-w-0 relative" onClick={(e) => e.stopPropagation()}>
        <img src={fullUrl} alt="" className="max-w-full max-h-full object-contain" draggable={false} />
        {index > 0 && (
          <button className="absolute left-2 top-1/2 -translate-y-1/2 w-10 h-10 flex items-center justify-center rounded-full bg-black/50 text-white/70 hover:text-white hover:bg-black/70 transition-colors" onClick={() => navigate(-1)}>
            <ChevronLeft size={24} />
          </button>
        )}
        {index < images.length - 1 && (
          <button className="absolute right-2 top-1/2 -translate-y-1/2 w-10 h-10 flex items-center justify-center rounded-full bg-black/50 text-white/70 hover:text-white hover:bg-black/70 transition-colors" onClick={() => navigate(1)}>
            <ChevronRight size={24} />
          </button>
        )}
        <div className="absolute bottom-2 left-1/2 -translate-x-1/2 text-3xs text-white/40 tabular-nums">
          {index + 1} / {images.length} | {img.width}x{img.height}
        </div>
      </div>

      {/* Metadata sidebar */}
      <div className="w-80 shrink-0 bg-background/95 border-l border-border/50 overflow-y-auto p-4 space-y-3" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium">Image Details</span>
          <div className="flex items-center gap-1">
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button size="sm" variant="ghost" className="h-7 px-2 text-xs gap-1">
                  <Send size={12} />Use
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-48">
                <DropdownMenuLabel className="text-3xs">Image</DropdownMenuLabel>
                <DropdownMenuItem onClick={async () => {
                  try {
                    const file = await fetchRemoteImage(fullUrl);
                    await sendImageToCanvas(file);
                    onCloseAll();
                    toast.success("Sent to Canvas");
                  } catch { toast.error("Failed to send image"); }
                }}>
                  <ImageIcon size={14} className="mr-2" />Send to Canvas
                </DropdownMenuItem>
                <DropdownMenuItem onClick={async () => {
                  try {
                    const file = await fetchRemoteImage(fullUrl);
                    const blob = new Blob([await file.arrayBuffer()], { type: file.type });
                    await sendFrameToVideoInit(blob);
                    onCloseAll();
                    toast.success("Sent to Video Init");
                  } catch { toast.error("Failed to send image"); }
                }}>
                  <ImageIcon size={14} className="mr-2" />Send to Video Init
                </DropdownMenuItem>
                {hasPrompt && (
                  <>
                    <DropdownMenuSeparator />
                    <DropdownMenuLabel className="text-3xs">Prompt</DropdownMenuLabel>
                    <DropdownMenuItem onClick={() => {
                      sendPromptToGeneration(meta.prompt as string, hasNegative ? meta.negativePrompt as string : undefined);
                      toast.success("Prompt sent to Generation");
                    }}>
                      <Sparkles size={14} className="mr-2" />Use in Generation
                    </DropdownMenuItem>
                    <DropdownMenuItem onClick={() => {
                      sendPromptToVideo(meta.prompt as string, hasNegative ? meta.negativePrompt as string : undefined);
                      toast.success("Prompt sent to Video");
                    }}>
                      <Sparkles size={14} className="mr-2" />Use in Video
                    </DropdownMenuItem>
                    <DropdownMenuSeparator />
                    <DropdownMenuItem onClick={async () => {
                      try {
                        const file = await fetchRemoteImage(fullUrl);
                        await sendImageToCanvas(file);
                        sendPromptToGeneration(meta.prompt as string, hasNegative ? meta.negativePrompt as string : undefined);
                        onCloseAll();
                        toast.success("Image + Prompt sent to Canvas");
                      } catch { toast.error("Failed to send"); }
                    }}>
                      <Send size={14} className="mr-2" />Image + Prompt → Canvas
                    </DropdownMenuItem>
                  </>
                )}
              </DropdownMenuContent>
            </DropdownMenu>
            <button className="w-7 h-7 flex items-center justify-center rounded text-muted-foreground hover:text-foreground hover:bg-muted/50 transition-colors" onClick={onClose}>
              <X size={16} />
            </button>
          </div>
        </div>
        {Object.keys(meta).length > 0 ? (
          <div className="space-y-2.5 pt-1">
            {META_DISPLAY_KEYS.map(([key, label]) => {
              const val = meta[key];
              if (val === undefined || val === null || val === "") return null;
              const isLong = typeof val === "string" && val.length > 60;
              const isPromptField = key === "prompt" || key === "negativePrompt";
              return (
                <div key={key}>
                  <div className="flex items-center justify-between mb-0.5">
                    <p className="text-3xs font-medium text-muted-foreground">{label}</p>
                    {isPromptField && (
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <button type="button" className="text-3xs text-primary hover:underline">Use</button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end" className="w-36">
                          <DropdownMenuItem onClick={() => {
                            sendPromptToGeneration(String(val));
                            toast.success(`${label} sent to Generation`);
                          }}>Generation</DropdownMenuItem>
                          <DropdownMenuItem onClick={() => {
                            sendPromptToVideo(String(val));
                            toast.success(`${label} sent to Video`);
                          }}>Video</DropdownMenuItem>
                        </DropdownMenuContent>
                      </DropdownMenu>
                    )}
                  </div>
                  {isLong ? (
                    <p className="text-xs text-foreground whitespace-pre-wrap break-words max-h-32 overflow-y-auto bg-muted/30 rounded p-2">{String(val)}</p>
                  ) : (
                    <p className="text-xs text-foreground">{String(val)}</p>
                  )}
                </div>
              );
            })}
          </div>
        ) : (
          <p className="text-xs text-muted-foreground/60 italic pt-2">No generation metadata available</p>
        )}
      </div>
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
  const [lightboxIndex, setLightboxIndex] = useState<number | null>(null);

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

  // Collect all non-video images from all versions for the hero strip
  // Limit to 10 per version — the version API only returns ~10 images with metadata,
  // so capping here ensures every displayed image can have generation params.
  const baseImages = useMemo(() => {
    if (!model) return [];
    const imgs: CivitImage[] = [];
    for (const v of model.modelVersions) {
      let count = 0;
      for (const img of v.images) {
        if (count >= 10) break;
        if (!img.url.toLowerCase().endsWith(".mp4")) {
          imgs.push(img);
          count++;
        }
      }
    }
    return imgs;
  }, [model]);

  // Collect version IDs that have images so we can fetch metadata per-version
  const versionIds = useMemo(() => {
    if (!model) return [];
    return model.modelVersions.filter((v) => v.images.length > 0).map((v) => v.id);
  }, [model]);

  // Fetch version details (which include images WITH meta) for all versions
  const { data: versionImages } = useCivitVersionImages(versionIds, versionIds.length > 0);

  // Merge metadata from version images into base images by UUID in URL
  const allImages = useMemo(() => {
    if (!versionImages?.length) return baseImages;
    const uuidRe = /\/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\//;
    const metaByUuid = new Map<string, Record<string, unknown>>();
    for (const img of versionImages) {
      if (img.meta && typeof img.meta === "object") {
        const m = uuidRe.exec(img.url);
        if (m) metaByUuid.set(m[1], img.meta as Record<string, unknown>);
      }
    }
    if (metaByUuid.size === 0) return baseImages;
    return baseImages.map((img) => {
      const m = uuidRe.exec(img.url);
      const meta = m ? metaByUuid.get(m[1]) : undefined;
      return meta ? { ...img, meta } : img;
    });
  }, [baseImages, versionImages]);

  const openLightbox = useCallback((thumbIndex: number) => {
    setLightboxIndex(thumbIndex);
  }, []);

  // State to hold the dialog portal container so the lightbox can be portalled inside it
  // (avoids Radix focus-trap/inert issues while escaping DialogContent CSS transforms)
  const [portalContainer, setPortalContainer] = useState<HTMLElement | null>(null);
  const capturePortal = useCallback((node: HTMLDivElement | null) => {
    if (node) {
      let el: HTMLElement | null = node;
      while (el && el.parentElement !== document.body) el = el.parentElement;
      setPortalContainer((prev) => prev === el ? prev : el);
    }
  }, []);

  return (
    <Dialog open={modelId !== null} onOpenChange={(open) => { if (!open) { if (lightboxIndex !== null) return; onClose(); } }}>
      <DialogContent showCloseButton={false} className="sm:max-w-5xl max-h-[85vh] flex flex-col p-0 gap-0 overflow-hidden" onPointerDownOutside={(e) => { if (lightboxIndex !== null) e.preventDefault(); }} onInteractOutside={(e) => { if (lightboxIndex !== null) e.preventDefault(); }}>
        <div ref={capturePortal} className="hidden" />
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
                    {model.tags.slice(0, 16).map((t, i) => (
                      <Badge key={`${t}-${i}`} variant="secondary" className="text-3xs px-1.5 py-0.5">{t}</Badge>
                    ))}
                  </div>
                )}

                {/* Image gallery */}
                <ImageStrip images={allImages} onImageClick={openLightbox} />

                {/* Model description */}
                {model.description && <HtmlDescription html={model.description} />}

                <Separator />

                {/* Versions */}
                <div className="space-y-3">
                  {model.modelVersions.map((v) => (
                    <VersionSection key={v.id} version={v} modelType={model.type} modelName={model.name} creatorName={model.creator.username} modelId={model.id} modelNsfw={model.nsfw} localFiles={localFiles} onImageClick={(img) => {
                      const idx = allImages.findIndex((a) => a.url === img.url);
                      if (idx >= 0) setLightboxIndex(idx);
                    }} />
                  ))}
                </div>
              </div>
            </div>

          </>
        ) : (
          <p className="text-sm text-muted-foreground py-8 text-center">Model not found</p>
        )}
      </DialogContent>

      {/* Image lightbox — portalled into dialog portal container to stay within Radix focus scope */}
      {lightboxIndex !== null && allImages.length > 0 && portalContainer && createPortal(
        <ImageLightbox images={allImages} index={lightboxIndex} onClose={() => setLightboxIndex(null)} onCloseAll={() => { setLightboxIndex(null); onClose(); }} onNavigate={setLightboxIndex} />,
        portalContainer,
      )}
    </Dialog>
  );
}
