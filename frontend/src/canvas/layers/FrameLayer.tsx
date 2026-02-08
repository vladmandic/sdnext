import { Layer, Rect, Label, Tag, Text } from "react-konva";
import { useGenerationStore } from "@/stores/generationStore";
import { contrastText } from "@/lib/utils";

const BORDER_COLOR = "#4ade80";
const LABEL_HEIGHT = 19; // fontSize(11) + padding(4)*2

export function FrameLayer() {
  const frameW = useGenerationStore((s) => s.width);
  const frameH = useGenerationStore((s) => s.height);

  return (
    <Layer listening={false}>
      <Rect
        x={0}
        y={0}
        width={frameW}
        height={frameH}
        stroke={BORDER_COLOR}
        strokeWidth={2}
        listening={false}
      />
      <Label x={0} y={-LABEL_HEIGHT} listening={false}>
        <Tag fill={BORDER_COLOR} cornerRadius={3} />
        <Text text="Input" fontSize={11} fill={contrastText(BORDER_COLOR)} padding={4} listening={false} />
      </Label>
    </Layer>
  );
}
