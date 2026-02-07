import { useEffect, useRef, useState } from "react";
import { Layer, Image as KonvaImage, Rect, Text } from "react-konva";
import { useGenerationStore } from "@/stores/generationStore";
import { base64ToObjectUrl } from "@/lib/utils";

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
  const objectUrlRef = useRef<string | null>(null);

  // Determine which image to show: live preview during generation, or selected result
  let displaySrc: string | undefined;
  if (previewImage) {
    displaySrc = previewImage;
  } else if (selectedResultId) {
    const selected = results.find((r) => r.id === selectedResultId);
    const raw = selected?.images[selectedImageIndex ?? 0];
    if (raw) {
      displaySrc = raw.startsWith("data:") || raw.startsWith("blob:") ? raw : base64ToObjectUrl(raw);
    }
  }

  useEffect(() => {
    // Revoke previous object URL to prevent leaks
    if (objectUrlRef.current) {
      URL.revokeObjectURL(objectUrlRef.current);
      objectUrlRef.current = null;
    }

    if (!displaySrc) {
      setImage(null);
      return;
    }

    // Track blob URLs we create for cleanup
    if (displaySrc.startsWith("blob:")) {
      objectUrlRef.current = displaySrc;
    }

    const img = new window.Image();
    img.onload = () => setImage(img);
    img.src = displaySrc;
  }, [displaySrc]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    };
  }, []);

  if (offsetX <= 0) return null;

  // Show image if available, otherwise dashed placeholder
  if (image) {
    return (
      <Layer>
        <KonvaImage image={image} x={offsetX} y={0} />
      </Layer>
    );
  }

  // Dashed placeholder rectangle
  return (
    <Layer>
      <Rect
        x={offsetX}
        y={0}
        width={placeholderWidth}
        height={placeholderHeight}
        stroke="#666"
        strokeWidth={1}
        dash={[8, 4]}
        cornerRadius={4}
      />
      <Text
        x={offsetX}
        y={placeholderHeight / 2 - 8}
        width={placeholderWidth}
        align="center"
        text="Output"
        fontSize={14}
        fill="#666"
      />
    </Layer>
  );
}
