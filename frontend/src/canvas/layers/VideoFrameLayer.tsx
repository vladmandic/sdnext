import { useEffect, useState, useCallback } from "react";
import { Layer, Group, Rect, Text, Image as KonvaImage } from "react-konva";
import { useVideoCanvasStore } from "@/stores/videoCanvasStore";

const ACTIVE_COLOR = "#4ade80";
const INACTIVE_COLOR = "#6b7280";

interface VideoFrameLayerProps {
  which: "init" | "last";
  offsetX: number;
  width: number;
  height: number;
  onPickImage?: () => void;
}

export function VideoFrameLayer({ which, offsetX, width, height, onPickImage }: VideoFrameLayerProps) {
  const frame = useVideoCanvasStore((s) => (which === "init" ? s.initFrame : s.lastFrame));
  const [image, setImage] = useState<HTMLImageElement | null>(null);

  const objectUrl = frame?.objectUrl;

  useEffect(() => {
    if (!objectUrl) return;
    const img = new window.Image();
    img.onload = () => setImage(img);
    img.src = objectUrl;
  }, [objectUrl]);

  // Clear stale image synchronously when frame removed
  if (!objectUrl && image) setImage(null);

  const hasImage = !!frame && !!image;

  const handleClick = useCallback((e: import("konva/lib/Node").KonvaEventObject<MouseEvent>) => {
    if (e.evt.button !== 0) return;
    if (!hasImage && onPickImage) onPickImage();
  }, [hasImage, onPickImage]);

  const handleTap = useCallback(() => {
    if (!hasImage && onPickImage) onPickImage();
  }, [hasImage, onPickImage]);

  return (
    <Layer>
      <Group x={offsetX}>
        {/* Dark background when empty — clickable */}
        {!hasImage && (
          <Rect
            x={0} y={0}
            width={width} height={height}
            fill="#1a1a1a"
            listening={true}
            onClick={handleClick}
            onTap={handleTap}
          />
        )}

        {/* Placeholder text */}
        {!hasImage && (
          <Text
            x={0}
            y={height / 2 - 8}
            width={width}
            align="center"
            text="Drop image or click"
            fontSize={14}
            fill="#666"
            listening={false}
          />
        )}

        {/* Image scaled to fit frame */}
        {hasImage && (
          <KonvaImage
            image={image}
            x={0} y={0}
            width={width} height={height}
            listening={false}
          />
        )}

        {/* Border */}
        <Rect
          x={0} y={0}
          width={width} height={height}
          stroke={hasImage ? ACTIVE_COLOR : INACTIVE_COLOR}
          strokeWidth={2}
          dash={hasImage ? undefined : [8, 4]}
          listening={false}
        />
      </Group>
    </Layer>
  );
}
