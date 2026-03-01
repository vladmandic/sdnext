import { useMemo } from "react";
import { createPortal } from "react-dom";
import type { GenerationResult } from "@/stores/generationStore";
import { resolveImageSrc } from "@/lib/utils";

interface ResultThumbPreviewProps {
  result: GenerationResult;
  imageIndex: number;
  anchorRect: DOMRect | null;
}

function parseInfoMeta(info: string): Record<string, string> {
  try {
    const parsed = JSON.parse(info);
    const meta: Record<string, string> = {};
    if (parsed.seed != null) meta.Seed = String(parsed.seed);
    if (parsed.steps != null) meta.Steps = String(parsed.steps);
    if (parsed.sampler_name) meta.Sampler = parsed.sampler_name;
    if (parsed.width && parsed.height) meta.Size = `${parsed.width}x${parsed.height}`;
    return meta;
  } catch {
    return {};
  }
}

export function ResultThumbPreview({ result, imageIndex, anchorRect }: ResultThumbPreviewProps) {
  const src = resolveImageSrc(result.images[imageIndex]);
  const meta = useMemo(() => parseInfoMeta(result.info), [result.info]);
  const entries = Object.entries(meta);

  if (!anchorRect) return null;

  const style: React.CSSProperties = {
    position: "fixed",
    left: anchorRect.left + anchorRect.width / 2,
    top: anchorRect.top - 8,
    transform: "translate(-50%, -100%)",
    zIndex: 60,
    pointerEvents: "none",
  };

  return createPortal(
    <div style={style} className="flex flex-col items-center">
      <div className="rounded-lg overflow-hidden border border-border bg-popover shadow-xl">
        <img src={src} alt="Preview" className="w-64 max-h-64 object-contain bg-black" />
        {entries.length > 0 && (
          <div className="px-2 py-1 flex flex-wrap gap-x-3 gap-y-0.5 text-3xs text-muted-foreground">
            {entries.map(([k, v]) => (
              <span key={k}><span className="text-foreground/60">{k}:</span> {v}</span>
            ))}
          </div>
        )}
      </div>
    </div>,
    document.body,
  );
}
