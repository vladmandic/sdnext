import { useCallback, useEffect, useMemo, useRef } from "react";
import { Layer, Line, Circle, Group, Image as KonvaImage } from "react-konva";
import { useCanvasStore } from "@/stores/canvasStore";
import { useImg2ImgStore } from "@/stores/img2imgStore";
import { useGenerationStore } from "@/stores/generationStore";
import type { MaskLine } from "@/stores/img2imgStore";
import type Konva from "konva";

interface MaskLayerProps {
  activeLineRef: React.RefObject<Konva.Line | null>;
  cursorRef: React.RefObject<Konva.Circle | null>;
}

/** Parse "#rrggbb" or "#rrggbbaa" into { rgb, alpha }. */
function parseMaskColor(color: string) {
  const rgb = color.slice(0, 7);
  const alpha = color.length > 7 ? parseInt(color.slice(7, 9), 16) / 255 : 1;
  return { rgb, alpha };
}

/**
 * Render committed mask strokes to an offscreen DOM canvas at full opacity.
 * Overlapping strokes merge naturally (red on red = red). Returns the canvas
 * element to be used as a Konva Image source displayed with a single alpha.
 */
function renderMaskCanvas(lines: MaskLine[], color: string, w: number, h: number): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d")!;

  for (const line of lines) {
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.lineWidth = line.strokeWidth;

    if (line.tool === "brush") {
      ctx.globalCompositeOperation = "source-over";
      ctx.strokeStyle = color;
    } else {
      ctx.globalCompositeOperation = "destination-out";
      ctx.strokeStyle = "#fff"; // color irrelevant, destination-out erases
    }

    ctx.beginPath();
    const pts = line.points;
    if (pts.length >= 2) {
      ctx.moveTo(pts[0], pts[1]);
      for (let i = 2; i < pts.length; i += 2) {
        ctx.lineTo(pts[i], pts[i + 1]);
      }
    }
    ctx.stroke();
  }

  return canvas;
}

/**
 * Committed strokes are rendered to a flat offscreen canvas (no alpha
 * compounding), then displayed as a single KonvaImage with the mask
 * alpha applied once. The active Line and cursor are controlled
 * imperatively by useMaskPaint — only committed-stroke changes cause
 * a React re-render.
 */
export function MaskLayer({ activeLineRef, cursorRef }: MaskLayerProps) {
  const maskVisible = useCanvasStore((s) => s.maskVisible);
  const maskColor = useCanvasStore((s) => s.maskColor);
  const maskLines = useImg2ImgStore((s) => s.maskLines);
  const frameW = useGenerationStore((s) => s.width);
  const frameH = useGenerationStore((s) => s.height);

  const { rgb, alpha } = parseMaskColor(maskColor);

  // Flatten committed strokes to a single canvas — overlaps merge
  const maskCanvas = useMemo(() => {
    if (maskLines.length === 0 || frameW <= 0 || frameH <= 0) return null;
    return renderMaskCanvas(maskLines, rgb, frameW, frameH);
  }, [maskLines, rgb, frameW, frameH]);

  // Sync the active line's stroke color when maskColor changes
  const activeLineNodeRef = useRef<Konva.Line | null>(null);
  const setActiveLineNode = useCallback((node: Konva.Line | null) => {
    activeLineNodeRef.current = node;
    (activeLineRef as React.MutableRefObject<Konva.Line | null>).current = node;
  }, [activeLineRef]);

  useEffect(() => {
    const node = activeLineNodeRef.current;
    if (node) {
      node.stroke(rgb);
      node.opacity(alpha);
    }
  }, [rgb, alpha]);

  const setCursorNode = useCallback((node: Konva.Circle | null) => {
    (cursorRef as React.MutableRefObject<Konva.Circle | null>).current = node;
  }, [cursorRef]);

  if (!maskVisible || frameW <= 0 || frameH <= 0) return null;

  return (
    <Layer>
      <Group clipFunc={(ctx) => { ctx.rect(0, 0, frameW, frameH); }}>
        {/* Committed strokes — flat canvas, single alpha, no compounding */}
        {maskCanvas && (
          <KonvaImage image={maskCanvas} x={0} y={0} opacity={alpha} />
        )}
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
        strokeWidth={1}
        dash={[4, 4]}
        visible={false}
        listening={false}
      />
    </Layer>
  );
}
