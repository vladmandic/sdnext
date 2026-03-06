import { useEffect, useState } from "react";
import { Layer, Image as KonvaImage, Rect } from "react-konva";
import { useGenerationStore } from "@/stores/generationStore";
import { useJobQueueStore, selectViewedJob } from "@/stores/jobStore";
import { resolveImageSrc } from "@/lib/utils";

const BORDER_COLOR = "#60a5fa";

interface OutputLayerProps {
  offsetX: number;
  placeholderWidth: number;
  placeholderHeight: number;
}

export function OutputLayer({ offsetX, placeholderWidth, placeholderHeight }: OutputLayerProps) {
  const viewedJob = useJobQueueStore(selectViewedJob);
  const previewImage = viewedJob?.previewUrl ?? null;
  const results = useGenerationStore((s) => s.results);
  const selectedResultId = useGenerationStore((s) => s.selectedResultId);
  const selectedImageIndex = useGenerationStore((s) => s.selectedImageIndex);
  const [image, setImage] = useState<HTMLImageElement | null>(null);

  // Determine which image to show: live preview during generation, or selected result
  let displaySrc: string | undefined;
  if (previewImage) {
    displaySrc = previewImage;
  } else if (selectedResultId) {
    const selected = results.find((r) => r.id === selectedResultId);
    const raw = selected?.images[selectedImageIndex ?? 0];
    if (raw) {
      displaySrc = resolveImageSrc(raw);
    }
  }

  useEffect(() => {
    if (!displaySrc) return;
    const img = new window.Image();
    img.onload = () => setImage(img);
    img.src = displaySrc;
  }, [displaySrc]);

  // Clear stale image when there's nothing to display
  if (!displaySrc && image) setImage(null);

  // Use displaySrc (synchronous) alongside image state so clearing is immediate
  const hasImage = !!displaySrc && !!image;
  const isPreview = !!previewImage;

  // Fit image inside the frame when aspect ratios differ (e.g. detailer preview
  // at 1024x1024 on a 16:9 output frame).  Frame never changes shape.
  let imgX = offsetX;
  let imgY = 0;
  let imgW = placeholderWidth;
  let imgH = placeholderHeight;
  if (isPreview && hasImage && image) {
    const natW = image.naturalWidth;
    const natH = image.naturalHeight;
    const frameAR = placeholderWidth / placeholderHeight;
    const imageAR = natW / natH;
    // Only use fit mode when aspect ratios meaningfully differ (>1% tolerance)
    if (Math.abs(frameAR - imageAR) / frameAR > 0.01) {
      if (imageAR > frameAR) {
        imgW = placeholderWidth;
        imgH = placeholderWidth / imageAR;
        imgY = (placeholderHeight - imgH) / 2;
      } else {
        imgH = placeholderHeight;
        imgW = placeholderHeight * imageAR;
        imgX = offsetX + (placeholderWidth - imgW) / 2;
      }
    }
  }

  return (
    <Layer listening={false}>
      {hasImage && (
        <KonvaImage
          image={image}
          x={imgX}
          y={imgY}
          width={imgW}
          height={imgH}
          opacity={isPreview ? 0.85 : 1}
        />
      )}
      <Rect
        x={offsetX}
        y={0}
        width={placeholderWidth}
        height={placeholderHeight}
        stroke={BORDER_COLOR}
        strokeWidth={2}
        dash={hasImage ? undefined : [8, 4]}
        listening={false}
      />
    </Layer>
  );
}
