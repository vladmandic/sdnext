import { useEffect, useState } from "react";
import { Layer, Image as KonvaImage, Rect } from "react-konva";
import { useGenerationStore } from "@/stores/generationStore";
import { resolveImageSrc } from "@/lib/utils";

const BORDER_COLOR = "#60a5fa";

interface OutputLayerProps {
  offsetX: number;
  placeholderWidth: number;
  placeholderHeight: number;
}

export function OutputLayer({ offsetX, placeholderWidth, placeholderHeight }: OutputLayerProps) {
  const previewImage = useGenerationStore((s) => s.previewImage);
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
  return (
    <Layer listening={false}>
      {hasImage && (
        <KonvaImage
          image={image}
          x={offsetX}
          y={0}
          width={placeholderWidth}
          height={placeholderHeight}
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
