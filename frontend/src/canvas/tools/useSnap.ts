import { useState, useMemo, useCallback, useRef, useEffect } from "react";
import { useCanvasStore } from "@/stores/canvasStore";
import type Konva from "konva";

const SNAP_THRESHOLD_PX = 8;

export interface Guide {
  orientation: "v" | "h";
  pos: number;
}

interface SnapResult {
  delta: number;
  snapped: number[];
}

function findBestSnap(edges: number[], targets: number[], threshold: number): SnapResult {
  let bestDelta = Infinity;
  const snapped: number[] = [];
  for (const edge of edges) {
    for (const target of targets) {
      const d = target - edge;
      if (Math.abs(d) < threshold && Math.abs(d) < Math.abs(bestDelta)) {
        bestDelta = d;
      }
    }
  }
  if (!isFinite(bestDelta)) return { delta: 0, snapped: [] };
  for (const edge of edges) {
    for (const target of targets) {
      if (Math.abs(target - (edge + bestDelta)) < 0.5) snapped.push(target);
    }
  }
  return { delta: bestDelta, snapped };
}

/** Anchor → which box edges are "free" (moving) during a transform */
const anchorFreeEdges: Record<string, { left?: boolean; right?: boolean; top?: boolean; bottom?: boolean }> = {
  "top-left": { left: true, top: true },
  "top-center": { top: true },
  "top-right": { right: true, top: true },
  "middle-left": { left: true },
  "middle-right": { right: true },
  "bottom-left": { left: true, bottom: true },
  "bottom-center": { bottom: true },
  "bottom-right": { right: true, bottom: true },
};

export function useSnap(frameW: number, frameH: number, trRef: React.RefObject<Konva.Transformer | null>, frameX = 0, frameY = 0, displayScale = 1) {
  const [guides, setGuides] = useState<Guide[]>([]);
  const scaleRef = useRef(1);
  const dsRef = useRef(displayScale);

  const viewportScale = useCanvasStore((s) => s.viewport.scale);
  useEffect(() => { scaleRef.current = viewportScale; }, [viewportScale]);
  useEffect(() => { dsRef.current = displayScale; }, [displayScale]);

  const xTargets = useMemo(() => [frameX, frameX + frameW / 2, frameX + frameW], [frameX, frameW]);
  const yTargets = useMemo(() => [frameY, frameY + frameH / 2, frameY + frameH], [frameY, frameH]);

  const handleDragMove = useCallback((e: Konva.KonvaEventObject<DragEvent>) => {
    const node = e.target as Konva.Node;
    const layer = node.getLayer();
    if (!layer) return;

    const ds = dsRef.current;
    const threshold = SNAP_THRESHOLD_PX / (scaleRef.current * ds);
    const rect = node.getClientRect({ relativeTo: layer });

    // getClientRect includes the displayScale Group transform — convert back to pixel space
    const xEdges = [rect.x / ds, (rect.x + rect.width / 2) / ds, (rect.x + rect.width) / ds];
    const yEdges = [rect.y / ds, (rect.y + rect.height / 2) / ds, (rect.y + rect.height) / ds];

    const xSnap = findBestSnap(xEdges, xTargets, threshold);
    const ySnap = findBestSnap(yEdges, yTargets, threshold);

    if (xSnap.delta) node.x(node.x() + xSnap.delta);
    if (ySnap.delta) node.y(node.y() + ySnap.delta);

    const newGuides: Guide[] = [];
    for (const pos of xSnap.snapped) newGuides.push({ orientation: "v", pos });
    for (const pos of ySnap.snapped) newGuides.push({ orientation: "h", pos });
    setGuides(newGuides);
  }, [xTargets, yTargets]);

  /** Snap during Transformer resize — adjusts node position/scale using actual visual bounds */
  const handleTransform = useCallback(() => {
    const tr = trRef.current;
    if (!tr) return;

    const anchor = tr.getActiveAnchor();
    if (!anchor) return;

    const free = anchorFreeEdges[anchor];
    if (!free) return;

    const node = tr.nodes()[0];
    if (!node) return;

    const layer = node.getLayer();
    if (!layer) return;

    const ds = dsRef.current;
    const threshold = SNAP_THRESHOLD_PX / (scaleRef.current * ds);
    const rect = node.getClientRect({ relativeTo: layer });
    // Convert display-space rect to pixel-space
    const pxRect = { x: rect.x / ds, y: rect.y / ds, width: rect.width / ds, height: rect.height / ds };
    const newGuides: Guide[] = [];

    // Right edge: adjust scaleX, left edge stays fixed
    if (free.right) {
      const rightEdge = pxRect.x + pxRect.width;
      for (const t of xTargets) {
        if (Math.abs(rightEdge - t) < threshold) {
          node.scaleX(node.scaleX() + (t - rightEdge) / node.width());
          newGuides.push({ orientation: "v", pos: t });
          break;
        }
      }
    }

    // Left edge: shift x and adjust scaleX to keep right edge fixed
    if (free.left) {
      const leftEdge = pxRect.x;
      for (const t of xTargets) {
        if (Math.abs(leftEdge - t) < threshold) {
          const dx = t - leftEdge;
          node.x(node.x() + dx);
          node.scaleX(node.scaleX() - dx / node.width());
          newGuides.push({ orientation: "v", pos: t });
          break;
        }
      }
    }

    // Bottom edge: adjust scaleY, top edge stays fixed
    if (free.bottom) {
      const bottomEdge = pxRect.y + pxRect.height;
      for (const t of yTargets) {
        if (Math.abs(bottomEdge - t) < threshold) {
          node.scaleY(node.scaleY() + (t - bottomEdge) / node.height());
          newGuides.push({ orientation: "h", pos: t });
          break;
        }
      }
    }

    // Top edge: shift y and adjust scaleY to keep bottom edge fixed
    if (free.top) {
      const topEdge = pxRect.y;
      for (const t of yTargets) {
        if (Math.abs(topEdge - t) < threshold) {
          const dy = t - topEdge;
          node.y(node.y() + dy);
          node.scaleY(node.scaleY() - dy / node.height());
          newGuides.push({ orientation: "h", pos: t });
          break;
        }
      }
    }

    setGuides(newGuides);
  }, [trRef, xTargets, yTargets]);

  const clearGuides = useCallback(() => setGuides([]), []);

  return { guides, handleDragMove, handleTransform, clearGuides };
}
