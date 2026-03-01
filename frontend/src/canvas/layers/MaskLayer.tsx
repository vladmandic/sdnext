import { useCallback, useEffect, useRef } from "react";
import { Layer, Line, Circle, Group } from "react-konva";
import { useCanvasStore } from "@/stores/canvasStore";
import { useGenerationStore } from "@/stores/generationStore";
import type Konva from "konva";

interface MaskLayerProps {
  displayScale: number;
  setActiveLineNode: (node: Konva.Line | null) => void;
  setCursorNode: (node: Konva.Circle | null) => void;
}

/** Parse "#rrggbb" or "#rrggbbaa" into { rgb, alpha }. */
function parseMaskColor(color: string) {
  const rgb = color.slice(0, 7);
  const alpha = color.length > 7 ? parseInt(color.slice(7, 9), 16) / 255 : 1;
  return { rgb, alpha };
}

/**
 * Active-stroke overlay for mask painting. Committed strokes are baked
 * into MaskObjectLayers (rendered in CompositeLayer). This layer only
 * shows the live stroke being drawn and the brush cursor.
 */
export function MaskLayer({ displayScale, setActiveLineNode: parentSetActiveLine, setCursorNode: parentSetCursor }: MaskLayerProps) {
  const maskVisible = useCanvasStore((s) => s.maskVisible);
  const maskColor = useCanvasStore((s) => s.maskColor);
  const frameW = useGenerationStore((s) => s.width);
  const frameH = useGenerationStore((s) => s.height);

  const { rgb, alpha } = parseMaskColor(maskColor);

  // Sync the active line's stroke color when maskColor changes
  const activeLineNodeRef = useRef<Konva.Line | null>(null);
  const setActiveLineNode = useCallback((node: Konva.Line | null) => {
    activeLineNodeRef.current = node;
    parentSetActiveLine(node);
  }, [parentSetActiveLine]);

  useEffect(() => {
    const node = activeLineNodeRef.current;
    if (node) {
      node.stroke(rgb);
      node.opacity(alpha);
    }
  }, [rgb, alpha]);

  const setCursorNode = useCallback((node: Konva.Circle | null) => {
    parentSetCursor(node);
  }, [parentSetCursor]);

  if (!maskVisible || frameW <= 0 || frameH <= 0) return null;

  return (
    <Layer listening={false}>
      <Group scaleX={displayScale} scaleY={displayScale}>
        <Group clipFunc={(ctx) => { ctx.rect(0, 0, frameW, frameH); }}>
          {/* Active stroke — drawn live by useMaskPaint */}
          <Line
            ref={setActiveLineNode}
            points={[]}
            stroke={rgb}
            strokeWidth={20}
            opacity={alpha}
            lineJoin="round"
            lineCap="round"
            visible={false}
            listening={false}
          />
        </Group>

        <Circle
          ref={setCursorNode}
          x={0}
          y={0}
          radius={10}
          stroke="#fff"
          strokeWidth={1 / displayScale}
          dash={[4 / displayScale, 4 / displayScale]}
          visible={false}
          listening={false}
        />
      </Group>
    </Layer>
  );
}
