import { useCallback } from "react";
import { Layer, Rect, Label, Tag, Text } from "react-konva";
import { useGenerationStore } from "@/stores/generationStore";
import { useCanvasStore } from "@/stores/canvasStore";
import { contrastText } from "@/lib/utils";

const BORDER_COLOR = "#4ade80";
const LABEL_HEIGHT = 19; // fontSize(11) + padding(4)*2

interface FrameLayerProps {
  onPickImage?: () => void;
}

export function FrameLayer({ onPickImage }: FrameLayerProps) {
  const frameW = useGenerationStore((s) => s.width);
  const frameH = useGenerationStore((s) => s.height);
  const hasLayers = useCanvasStore((s) => s.layers.length > 0);

  const handleClick = useCallback(() => {
    if (!hasLayers && onPickImage) onPickImage();
  }, [hasLayers, onPickImage]);

  return (
    <Layer>
      {/* Clickable background when empty */}
      {!hasLayers && (
        <Rect
          x={0}
          y={0}
          width={frameW}
          height={frameH}
          fill="transparent"
          listening={true}
          onClick={handleClick}
          onTap={handleClick}
        />
      )}

      {/* Placeholder text when no images */}
      {!hasLayers && (
        <Text
          x={0}
          y={frameH / 2 - 8}
          width={frameW}
          align="center"
          text="Drop image or click to upload"
          fontSize={14}
          fill="#666"
          listening={false}
        />
      )}

      {/* Border — dashed when empty, solid when has layers */}
      <Rect
        x={0}
        y={0}
        width={frameW}
        height={frameH}
        stroke={BORDER_COLOR}
        strokeWidth={2}
        dash={hasLayers ? undefined : [8, 4]}
        listening={false}
      />

      {/* Label */}
      <Label x={0} y={-LABEL_HEIGHT} listening={false}>
        <Tag fill={BORDER_COLOR} cornerRadius={3} />
        <Text text="Input" fontSize={11} fill={contrastText(BORDER_COLOR)} padding={4} listening={false} />
      </Label>
    </Layer>
  );
}
