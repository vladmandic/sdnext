import { useEffect, useRef, useState } from "react";
import { Layer, Rect, Image as KonvaImage } from "react-konva";
import { useControlStore } from "@/stores/controlStore";

const BORDER_COLOR = "#c084fc"; // purple-400

interface ProcessedCompositeLayerProps {
  offsetX: number;
  width: number;
  height: number;
}

export function ProcessedCompositeLayer({ offsetX, width, height }: ProcessedCompositeLayerProps) {
  const units = useControlStore((s) => s.units);
  const compositeProcessed = useControlStore((s) => s.compositeProcessed);
  const [displayImage, setDisplayImage] = useState<HTMLImageElement | HTMLCanvasElement | null>(null);
  const prevKeyRef = useRef<string>("");

  // Collect per-unit processed data URLs (from manual preprocessing)
  const perUnitSrcs = units
    .filter((u) => u.enabled && !!u.processedImage)
    .map((u) => u.processedImage!);

  // Fingerprint: prefer backend composite, fall back to per-unit
  const srcsKey = compositeProcessed ?? perUnitSrcs.join("\n");

  useEffect(() => {
    if (srcsKey === prevKeyRef.current) return;
    prevKeyRef.current = srcsKey;

    if (!srcsKey) {
      setDisplayImage(null);
      return;
    }

    const aborted = { current: false };

    const load = async () => {
      if (compositeProcessed) {
        // Backend composite: load single image directly
        const img = new window.Image();
        img.src = compositeProcessed;
        await new Promise<void>((resolve) => { img.onload = () => resolve(); img.onerror = () => resolve(); });
        if (!aborted.current && img.naturalWidth > 0) setDisplayImage(img);
        else if (!aborted.current) setDisplayImage(null);
        return;
      }

      // Per-unit compositing (manual preprocessing)
      const images: HTMLImageElement[] = [];
      for (const src of perUnitSrcs) {
        if (aborted.current) return;
        const img = new window.Image();
        img.src = src;
        await new Promise<void>((resolve) => { img.onload = () => resolve(); img.onerror = () => resolve(); });
        if (aborted.current) return;
        if (img.naturalWidth > 0) images.push(img);
      }

      if (aborted.current || images.length === 0) {
        if (!aborted.current) setDisplayImage(null);
        return;
      }

      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      ctx.globalCompositeOperation = "lighter";
      for (const img of images) {
        ctx.drawImage(img, 0, 0, width, height);
      }

      if (!aborted.current) setDisplayImage(canvas);
    };

    load();
    return () => { aborted.current = true; };
  // eslint-disable-next-line react-hooks/exhaustive-deps -- srcsKey covers all source changes; width/height for canvas size
  }, [srcsKey, width, height]);

  return (
    <Layer>
      {displayImage && (
        <KonvaImage image={displayImage} x={offsetX} y={0} width={width} height={height} listening={false} />
      )}
      <Rect
        x={offsetX}
        y={0}
        width={width}
        height={height}
        stroke={BORDER_COLOR}
        strokeWidth={2}
        dash={displayImage ? undefined : [8, 4]}
        listening={false}
      />
    </Layer>
  );
}
