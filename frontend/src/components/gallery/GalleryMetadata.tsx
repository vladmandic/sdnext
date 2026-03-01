import { useMemo } from "react";
import { useGalleryStore } from "@/stores/galleryStore";
import { useGenerationStore } from "@/stores/generationStore";
import { parseGenerationInfo } from "@/lib/parseGenerationInfo";
import { isVideoFile } from "@/lib/mediaType";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Button } from "@/components/ui/button";
import { Copy, Download, ExternalLink, ImageIcon } from "lucide-react";

interface VideoMeta {
  codec: string;
  frames: string;
  duration: string;
  fps: string;
}

function parseVideoExif(exif: string): VideoMeta | null {
  if (!exif) return null;
  const result: VideoMeta = { codec: "", frames: "", duration: "", fps: "" };
  let matched = false;
  for (const part of exif.split(",")) {
    const trimmed = part.trim();
    const [key, ...rest] = trimmed.split(":");
    const val = rest.join(":").trim();
    const k = key.trim().toLowerCase();
    if (k === "codec") { result.codec = val; matched = true; }
    else if (k === "frames") { result.frames = val; matched = true; }
    else if (k === "duration") { result.duration = val; matched = true; }
    else if (k === "fps") { result.fps = val; matched = true; }
  }
  return matched ? result : null;
}

export function GalleryMetadata() {
  const selectedFile = useGalleryStore((s) => s.selectedFile);
  const selectedThumb = useGalleryStore((s) => s.selectedThumb);

  const isVideo = selectedFile ? isVideoFile(selectedFile.relativePath) : false;

  const genInfo = useMemo(() => {
    return parseGenerationInfo(selectedThumb?.exif);
  }, [selectedThumb]);

  const videoMeta = useMemo(() => {
    if (!isVideo || !selectedThumb?.exif) return null;
    return parseVideoExif(selectedThumb.exif);
  }, [isVideo, selectedThumb]);

  if (!selectedFile || !selectedThumb) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-muted-foreground text-xs p-3">
        <ImageIcon size={20} className="opacity-30 mb-2" />
        <p>Select an image to view details</p>
      </div>
    );
  }

  const filename = selectedFile.relativePath.split("/").pop() ?? selectedFile.relativePath;
  const fullUrl = `/file=${selectedFile.fullPath}`;

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const formatDate = (ms: number) => {
    return new Date(ms).toLocaleString();
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text).catch(() => {});
  };

  const copyPromptToGeneration = () => {
    if (genInfo.prompt) {
      useGenerationStore.getState().setParam("prompt", genInfo.prompt);
    }
  };

  const copyNegativeToGeneration = () => {
    if (genInfo.negativePrompt) {
      useGenerationStore.getState().setParam("negativePrompt", genInfo.negativePrompt);
    }
  };

  return (
    <ScrollArea className="h-full">
      <div className="p-3 space-y-3">
        {/* Preview */}
        <div className="rounded-md overflow-hidden border border-border bg-muted">
          {isVideo ? (
            <video
              src={fullUrl}
              poster={selectedThumb.data}
              controls
              muted
              className="w-full object-contain max-h-48"
            />
          ) : (
            <img
              src={selectedThumb.data}
              alt={filename}
              className="w-full object-contain max-h-48"
            />
          )}
        </div>

        {/* File info */}
        <div>
          <h3 className="text-3xs font-semibold text-muted-foreground uppercase tracking-wider mb-1.5">File</h3>
          <div className="space-y-1">
            <MetaRow label="Name" value={filename} />
            <MetaRow label="Size" value={formatSize(selectedThumb.size)} />
            <MetaRow label="Dimensions" value={`${selectedThumb.width} x ${selectedThumb.height}`} />
            <MetaRow label="Modified" value={formatDate(selectedThumb.mtime)} />
            <MetaRow label="Path" value={selectedFile.relativePath} />
          </div>
        </div>

        {/* Video metadata */}
        {videoMeta && (
          <>
            <Separator />
            <div>
              <h3 className="text-3xs font-semibold text-muted-foreground uppercase tracking-wider mb-1.5">Video</h3>
              <div className="space-y-1">
                {videoMeta.duration && <MetaRow label="Duration" value={videoMeta.duration} />}
                {videoMeta.fps && <MetaRow label="FPS" value={videoMeta.fps} />}
                {videoMeta.frames && <MetaRow label="Frames" value={videoMeta.frames} />}
                {videoMeta.codec && <MetaRow label="Codec" value={videoMeta.codec} />}
              </div>
            </div>
          </>
        )}

        {/* Image generation info (hidden for videos) */}
        {!isVideo && genInfo.prompt && (
          <>
            <Separator />
            <div>
              <div className="flex items-center justify-between mb-1.5">
                <h3 className="text-3xs font-semibold text-muted-foreground uppercase tracking-wider">Prompt</h3>
                <div className="flex gap-0.5">
                  <SmallButton onClick={() => copyToClipboard(genInfo.prompt)} title="Copy"><Copy size={10} /></SmallButton>
                  <SmallButton onClick={copyPromptToGeneration} title="Use">Use</SmallButton>
                </div>
              </div>
              <p className="text-2xs text-foreground leading-relaxed break-words">{genInfo.prompt}</p>
            </div>
          </>
        )}

        {!isVideo && genInfo.negativePrompt && (
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <h3 className="text-3xs font-semibold text-muted-foreground uppercase tracking-wider">Negative</h3>
              <div className="flex gap-0.5">
                <SmallButton onClick={() => copyToClipboard(genInfo.negativePrompt)} title="Copy"><Copy size={10} /></SmallButton>
                <SmallButton onClick={copyNegativeToGeneration} title="Use">Use</SmallButton>
              </div>
            </div>
            <p className="text-2xs text-foreground/70 leading-relaxed break-words">{genInfo.negativePrompt}</p>
          </div>
        )}

        {!isVideo && Object.keys(genInfo.params).length > 0 && (
          <>
            <Separator />
            <div>
              <h3 className="text-3xs font-semibold text-muted-foreground uppercase tracking-wider mb-1.5">Parameters</h3>
              <div className="space-y-0.5">
                {Object.entries(genInfo.params).map(([key, value]) => (
                  <MetaRow key={key} label={key} value={value} />
                ))}
              </div>
            </div>
          </>
        )}

        <Separator />

        {/* Actions */}
        <div className="space-y-1.5">
          <Button variant="outline" size="sm" className="w-full h-6 text-2xs justify-start gap-2" asChild>
            <a href={fullUrl} target="_blank" rel="noopener noreferrer">
              <ExternalLink size={12} /> Open full size
            </a>
          </Button>
          <Button variant="outline" size="sm" className="w-full h-6 text-2xs justify-start gap-2" asChild>
            <a href={fullUrl} download={filename}>
              <Download size={12} /> Download
            </a>
          </Button>
        </div>
      </div>
    </ScrollArea>
  );
}

function MetaRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-start gap-2">
      <span className="text-3xs text-muted-foreground w-16 flex-shrink-0">{label}</span>
      <span className="text-2xs text-foreground break-all">{value}</span>
    </div>
  );
}

function SmallButton({ children, onClick, title }: { children: React.ReactNode; onClick: () => void; title?: string }) {
  return (
    <button
      onClick={onClick}
      title={title}
      className="px-1.5 py-0.5 text-4xs rounded bg-accent/50 text-accent-foreground hover:bg-accent transition-colors"
    >
      {children}
    </button>
  );
}
