import { useEffect, useRef, useState } from "react";
import { Layer, Image as KonvaImage, Rect, Label, Tag, Text } from "react-konva";
import { useGenerationStore } from "@/stores/generationStore";
import { base64ToObjectUrl, contrastText } from "@/lib/utils";

const BORDER_COLOR = "#60a5fa";
const LABEL_HEIGHT = 19; // fontSize(11) + padding(4)*2

interface OutputLayerProps {
  offsetX: number;
  placeholderWidth: number;
  placeholderHeight: number;
}

export function OutputLayer({ offsetX, placeholderWidth, placeholderHeight }: OutputLayerProps) {
  const isGenerating = useGenerationStore((s) => s.isGenerating);
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

    if (!displaySrc) return;

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

  // Use displaySrc (synchronous) alongside image state so clearing is immediate
  const hasImage = !!displaySrc && !!image;
  const isLivePreview = isGenerating && !!previewImage;

  // Frame stays at placeholder (= input) size during previews and when empty.
  // Only adjusts to actual image dimensions for a finished result.
  const frameW = hasImage && !isLivePreview ? image.naturalWidth : placeholderWidth;
  const frameH = hasImage && !isLivePreview ? image.naturalHeight : placeholderHeight;

  return (
    <Layer listening={false}>
      {hasImage ? (
        <KonvaImage
          image={image}
          x={offsetX}
          y={0}
          width={isLivePreview ? placeholderWidth : undefined}
          height={isLivePreview ? placeholderHeight : undefined}
        />
      ) : (
        <Text
          x={offsetX}
          y={placeholderHeight / 2 - 8}
          width={placeholderWidth}
          align="center"
          text="Output"
          fontSize={14}
          fill="#666"
          listening={false}
        />
      )}
      <Rect
        x={offsetX}
        y={0}
        width={frameW}
        height={frameH}
        stroke={BORDER_COLOR}
        strokeWidth={2}
        dash={hasImage ? undefined : [8, 4]}
        listening={false}
      />
      <Label x={offsetX} y={-LABEL_HEIGHT} listening={false}>
        <Tag fill={BORDER_COLOR} cornerRadius={3} />
        <Text text="Output" fontSize={11} fill={contrastText(BORDER_COLOR)} padding={4} listening={false} />
      </Label>
    </Layer>
  );
}
