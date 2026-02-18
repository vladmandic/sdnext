import { useEffect, useRef, useCallback } from "react";
import { Layer, Image as KonvaImage, Transformer, Line } from "react-konva";
import { useCanvasStore, type ImageLayer as ImageLayerType } from "@/stores/canvasStore";
import { useGenerationStore } from "@/stores/generationStore";
import { useSnap } from "@/canvas/tools/useSnap";
import type Konva from "konva";

interface CompositeLayerProps {
  trRef: React.RefObject<Konva.Transformer | null>;
}

export function CompositeLayer({ trRef }: CompositeLayerProps) {
  const layers = useCanvasStore((s) => s.layers);
  const activeLayerId = useCanvasStore((s) => s.activeLayerId);
  const activeTool = useCanvasStore((s) => s.activeTool);
  const updateLayer = useCanvasStore((s) => s.updateLayer);
  const setActiveLayer = useCanvasStore((s) => s.setActiveLayer);
  const frameW = useGenerationStore((s) => s.width);
  const frameH = useGenerationStore((s) => s.height);
  const snap = useSnap(frameW, frameH, trRef);

  const imageMap = useRef<Map<string, HTMLImageElement>>(new Map());
  const nodeMap = useRef<Map<string, Konva.Image>>(new Map());

  const imageLayers = layers.filter((l) => l.type === "image") as ImageLayerType[];

  // Load/unload HTMLImageElements as layers change
  useEffect(() => {
    const current = new Set<string>();
    for (const layer of imageLayers) {
      current.add(layer.id);
      if (!imageMap.current.has(layer.id)) {
        const img = new window.Image();
        img.src = layer.imageData;
        imageMap.current.set(layer.id, img);
      }
    }
    // Clean up removed layers
    for (const id of imageMap.current.keys()) {
      if (!current.has(id)) {
        imageMap.current.delete(id);
        nodeMap.current.delete(id);
      }
    }
  }, [imageLayers]);

  // Attach transformer to active layer node
  useEffect(() => {
    if (!trRef.current) return;
    if (activeLayerId && activeTool === "move") {
      const node = nodeMap.current.get(activeLayerId);
      if (node) {
        trRef.current.nodes([node]);
        trRef.current.getLayer()?.batchDraw();
        return;
      }
    }
    trRef.current.nodes([]);
    trRef.current.getLayer()?.batchDraw();
  }, [activeLayerId, activeTool, trRef, imageLayers]);

  const handleDragEnd = useCallback((layerId: string, e: Konva.KonvaEventObject<DragEvent>) => {
    snap.clearGuides();
    updateLayer(layerId, {
      x: e.target.x(),
      y: e.target.y(),
    } as Partial<ImageLayerType>);
  }, [updateLayer, snap]);

  const handleTransformEnd = useCallback((layerId: string, e: Konva.KonvaEventObject<Event>) => {
    snap.clearGuides();
    const node = e.target as Konva.Image;
    updateLayer(layerId, {
      x: node.x(),
      y: node.y(),
      scaleX: node.scaleX(),
      scaleY: node.scaleY(),
      rotation: node.rotation(),
    } as Partial<ImageLayerType>);
  }, [updateLayer, snap]);

  const handleClick = useCallback((layerId: string, e: Konva.KonvaEventObject<MouseEvent>) => {
    if (e.evt.button !== 0 || activeTool !== "move") return;
    e.cancelBubble = true;
    setActiveLayer(layerId);
  }, [activeTool, setActiveLayer]);

  const setNodeRef = useCallback((layerId: string, node: Konva.Image | null) => {
    if (node) {
      nodeMap.current.set(layerId, node);
    } else {
      nodeMap.current.delete(layerId);
    }
  }, []);

  return (
    <Layer>
      {/* eslint-disable-next-line react-hooks/refs -- imageMap synced with imageLayers in effect above */}
      {imageLayers.map((layer) => (
        <KonvaImage
          key={layer.id}
          ref={(node) => setNodeRef(layer.id, node)}
          image={imageMap.current.get(layer.id)}
          x={layer.x}
          y={layer.y}
          scaleX={layer.scaleX}
          scaleY={layer.scaleY}
          rotation={layer.rotation}
          opacity={layer.opacity}
          visible={layer.visible}
          draggable={activeTool === "move" && !layer.locked}
          onDragMove={snap.handleDragMove}
          onDragEnd={(e) => handleDragEnd(layer.id, e)}
          onTransformEnd={(e) => handleTransformEnd(layer.id, e)}
          onClick={(e) => handleClick(layer.id, e)}
        />
      ))}
      <Transformer
        ref={trRef}
        keepRatio={false}
        enabledAnchors={[
          "top-left", "top-right", "bottom-left", "bottom-right",
          "top-center", "bottom-center", "middle-left", "middle-right",
        ]}
        onTransform={snap.handleTransform}
      />
      {snap.guides.map((g, i) => (
        <Line
          key={i}
          points={g.orientation === "v" ? [g.pos, -5000, g.pos, 5000] : [-5000, g.pos, 5000, g.pos]}
          stroke="#22d3ee"
          strokeWidth={1}
          strokeScaleEnabled={false}
          listening={false}
        />
      ))}
    </Layer>
  );
}
