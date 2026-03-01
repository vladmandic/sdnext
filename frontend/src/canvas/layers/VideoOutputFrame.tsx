import { useEffect, useState } from "react";
import { Layer, Image as KonvaImage, Rect } from "react-konva";
import { useJobQueueStore, selectVideoDomainActiveJob } from "@/stores/jobStore";
import { useVideoStore } from "@/stores/videoStore";

const BORDER_COLOR = "#60a5fa";

interface VideoOutputFrameProps {
  offsetX: number;
  width: number;
  height: number;
}

export function VideoOutputFrame({ offsetX, width, height }: VideoOutputFrameProps) {
  const activeJob = useJobQueueStore(selectVideoDomainActiveJob);
  const previewUrl = activeJob?.previewUrl ?? null;
  const selectedResultId = useVideoStore((s) => s.selectedResultId);
  const [image, setImage] = useState<HTMLImageElement | null>(null);

  // Show preview image during generation on the Konva canvas
  useEffect(() => {
    if (!previewUrl) return;
    const img = new window.Image();
    img.onload = () => setImage(img);
    img.src = previewUrl;
  }, [previewUrl]);

  // Clear stale image synchronously when preview gone
  if (!previewUrl && image) setImage(null);

  const hasPreview = !!previewUrl && !!image;
  const hasResult = !!selectedResultId;

  return (
    <Layer listening={false}>
      {hasPreview && (
        <KonvaImage
          image={image}
          x={offsetX} y={0}
          width={width} height={height}
        />
      )}
      <Rect
        x={offsetX} y={0}
        width={width} height={height}
        stroke={BORDER_COLOR}
        strokeWidth={2}
        dash={(hasPreview || hasResult) ? undefined : [8, 4]}
        listening={false}
      />
    </Layer>
  );
}
