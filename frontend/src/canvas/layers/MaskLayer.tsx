import { Layer, Line, Circle, Group } from "react-konva";
import { useCanvasStore } from "@/stores/canvasStore";
import { useImg2ImgStore } from "@/stores/img2imgStore";
import type { MaskLine } from "@/stores/img2imgStore";

interface MaskLayerProps {
  currentLine: MaskLine | null;
  cursorPos: { x: number; y: number } | null;
}

function MaskStroke({ line, color }: { line: MaskLine; color: string }) {
  return (
    <Line
      points={line.points}
      stroke={color}
      strokeWidth={line.strokeWidth}
      lineJoin="round"
      lineCap="round"
      globalCompositeOperation={line.tool === "eraser" ? "destination-out" : "source-over"}
    />
  );
}

export function MaskLayer({ currentLine, cursorPos }: MaskLayerProps) {
  const maskVisible = useCanvasStore((s) => s.maskVisible);
  const maskColor = useCanvasStore((s) => s.maskColor);
  const activeTool = useCanvasStore((s) => s.activeTool);
  const brushSize = useCanvasStore((s) => s.brushSize);
  const viewport = useCanvasStore((s) => s.viewport);
  const maskLines = useImg2ImgStore((s) => s.maskLines);
  const initW = useImg2ImgStore((s) => s.initImageWidth);
  const initH = useImg2ImgStore((s) => s.initImageHeight);

  if (!maskVisible || initW <= 0 || initH <= 0) return null;

  const isMaskTool = activeTool === "maskBrush" || activeTool === "maskEraser";
  const hasStrokes = maskLines.length > 0 || currentLine !== null;

  return (
    <Layer>
      {/* Mask strokes clipped to init image bounds */}
      {hasStrokes && (
        <Group
          clipFunc={(ctx) => {
            ctx.rect(0, 0, initW, initH);
          }}
        >
          {maskLines.map((line, i) => (
            <MaskStroke key={i} line={line} color={maskColor} />
          ))}
          {currentLine && <MaskStroke line={currentLine} color={maskColor} />}
        </Group>
      )}

      {/* Brush cursor */}
      {isMaskTool && cursorPos && (
        <Circle
          x={cursorPos.x}
          y={cursorPos.y}
          radius={brushSize / 2}
          stroke="#fff"
          strokeWidth={1 / viewport.scale}
          dash={[4 / viewport.scale, 4 / viewport.scale]}
          listening={false}
        />
      )}
    </Layer>
  );
}
