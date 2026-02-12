import { useEffect, useRef, useState, useCallback } from "react";
import { Layer, Rect, Label, Tag, Text, Image as KonvaImage } from "react-konva";
import { useControlStore } from "@/stores/controlStore";
import { useCanvasStore } from "@/stores/canvasStore";
import { contrastText } from "@/lib/utils";
import { PROCESSED_GAP, type ControlFramePosition } from "@/canvas/useControlFrameLayout";

const BORDER_COLOR = "#f59e0b"; // amber
const PROCESSED_BORDER_COLOR = "#78716c"; // stone-500
const LABEL_HEIGHT = 19;

interface ControlFrameLayerProps {
  frames: ControlFramePosition[];
  onPickImage?: (unitIndex: number) => void;
}

interface FrameImageState {
  objectUrl: string;
  htmlImage: HTMLImageElement;
}

interface ProcessedImageState {
  htmlImage: HTMLImageElement;
}

export function ControlFrameLayer({ frames, onPickImage }: ControlFrameLayerProps) {
  const units = useControlStore((s) => s.units);
  const setSelectedControlFrame = useCanvasStore((s) => s.setSelectedControlFrame);

  // Track loaded images per unit index
  const [imageMap, setImageMap] = useState<Map<number, FrameImageState>>(new Map());
  const [processedMap, setProcessedMap] = useState<Map<number, ProcessedImageState>>(new Map());
  const prevUrlsRef = useRef<Map<number, string>>(new Map());
  const prevProcessedUrlsRef = useRef<Map<number, string>>(new Map());

  // Sync image state with controlStore units
  useEffect(() => {
    const newPrevUrls = new Map<number, string>();
    const toRevoke: string[] = [];
    const toLoad: { index: number; file: File }[] = [];

    for (const frame of frames) {
      const unit = units[frame.unitIndex];
      const file = unit?.image;
      const prevUrl = prevUrlsRef.current.get(frame.unitIndex);

      if (file) {
        // Check if the File reference changed
        const existing = imageMap.get(frame.unitIndex);
        if (existing && prevUrl) {
          // Same object URL means same file — skip
          newPrevUrls.set(frame.unitIndex, prevUrl);
        } else {
          toLoad.push({ index: frame.unitIndex, file });
        }
      } else if (prevUrl) {
        toRevoke.push(prevUrl);
      }
    }

    // Revoke old URLs for removed images
    for (const url of toRevoke) URL.revokeObjectURL(url);

    if (toLoad.length === 0 && toRevoke.length === 0) return;

    // Load new images
    const aborted = { current: false };
    const loadImages = async () => {
      const newEntries = new Map(imageMap);
      for (const { index, file } of toLoad) {
        if (aborted.current) return;
        const url = URL.createObjectURL(file);
        newPrevUrls.set(index, url);
        const img = new window.Image();
        img.src = url;
        await new Promise<void>((resolve) => { img.onload = () => resolve(); img.onerror = () => resolve(); });
        if (aborted.current) { URL.revokeObjectURL(url); return; }
        newEntries.set(index, { objectUrl: url, htmlImage: img });
      }

      // Clean entries for frames that no longer exist
      for (const key of newEntries.keys()) {
        if (!frames.some((f) => f.unitIndex === key)) {
          const entry = newEntries.get(key);
          if (entry) URL.revokeObjectURL(entry.objectUrl);
          newEntries.delete(key);
        }
      }

      setImageMap(newEntries);
      prevUrlsRef.current = newPrevUrls;
    };
    loadImages();

    return () => { aborted.current = true; };
  // eslint-disable-next-line react-hooks/exhaustive-deps -- intentionally tracking specific refs
  }, [frames, units]);

  // Sync processed images
  useEffect(() => {
    const toLoad: { index: number; src: string }[] = [];

    for (const frame of frames) {
      const unit = units[frame.unitIndex];
      const src = unit?.processedImage;
      const prevSrc = prevProcessedUrlsRef.current.get(frame.unitIndex);

      if (src && src !== prevSrc) {
        toLoad.push({ index: frame.unitIndex, src });
      } else if (!src && prevSrc) {
        setProcessedMap((prev) => {
          const next = new Map(prev);
          next.delete(frame.unitIndex);
          return next;
        });
      }
    }

    if (toLoad.length === 0) return;

    const aborted = { current: false };
    const loadProcessed = async () => {
      const newProcessed = new Map(processedMap);
      const newPrevUrls = new Map(prevProcessedUrlsRef.current);
      for (const { index, src } of toLoad) {
        if (aborted.current) return;
        const img = new window.Image();
        img.src = src;
        await new Promise<void>((resolve) => { img.onload = () => resolve(); img.onerror = () => resolve(); });
        if (aborted.current) return;
        newProcessed.set(index, { htmlImage: img });
        newPrevUrls.set(index, src);
      }
      setProcessedMap(newProcessed);
      prevProcessedUrlsRef.current = newPrevUrls;
    };
    loadProcessed();

    return () => { aborted.current = true; };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [frames, units]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      for (const entry of imageMap.values()) URL.revokeObjectURL(entry.objectUrl);
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleFrameClick = useCallback((unitIndex: number, hasImage: boolean) => {
    if (!hasImage && onPickImage) {
      onPickImage(unitIndex);
    } else {
      setSelectedControlFrame(unitIndex);
    }
  }, [setSelectedControlFrame, onPickImage]);

  if (frames.length === 0) return null;

  return (
    <Layer>
      {frames.map((frame) => {
        const unit = units[frame.unitIndex];
        const imgState = imageMap.get(frame.unitIndex);
        const processedState = processedMap.get(frame.unitIndex);
        const hasImage = !!imgState;
        const labelText = `Control ${frame.unitIndex} (${unit?.unitType ?? "?"})`;

        return (
          <ControlFrame
            key={frame.unitIndex}
            frame={frame}
            labelText={labelText}
            hasImage={hasImage}
            image={imgState?.htmlImage ?? null}
            processedImage={processedState?.htmlImage ?? null}
            onClick={handleFrameClick}
          />
        );
      })}
    </Layer>
  );
}

interface ControlFrameProps {
  frame: ControlFramePosition;
  labelText: string;
  hasImage: boolean;
  image: HTMLImageElement | null;
  processedImage: HTMLImageElement | null;
  onClick: (unitIndex: number, hasImage: boolean) => void;
}

function ControlFrame({ frame, labelText, hasImage, image, processedImage, onClick }: ControlFrameProps) {
  const handleClick = useCallback(() => {
    onClick(frame.unitIndex, hasImage);
  }, [onClick, frame.unitIndex, hasImage]);

  const processedY = frame.y + frame.height + PROCESSED_GAP;

  return (
    <>
      {/* Clickable background rect */}
      <Rect
        x={frame.x}
        y={frame.y}
        width={frame.width}
        height={frame.height}
        fill="transparent"
        listening={true}
        onClick={handleClick}
        onTap={handleClick}
        name="controlFrame"
      />

      {/* Image or placeholder */}
      {hasImage && image ? (
        <KonvaImage
          image={image}
          x={frame.x}
          y={frame.y}
          width={frame.width}
          height={frame.height}
          listening={false}
        />
      ) : (
        <Text
          x={frame.x}
          y={frame.y + frame.height / 2 - 8}
          width={frame.width}
          align="center"
          text="Drop image or click to upload"
          fontSize={14}
          fill="#666"
          listening={false}
        />
      )}

      {/* Border */}
      <Rect
        x={frame.x}
        y={frame.y}
        width={frame.width}
        height={frame.height}
        stroke={BORDER_COLOR}
        strokeWidth={2}
        dash={hasImage ? undefined : [8, 4]}
        listening={false}
      />

      {/* Label */}
      <Label x={frame.x} y={-LABEL_HEIGHT} listening={false}>
        <Tag fill={BORDER_COLOR} cornerRadius={3} />
        <Text text={labelText} fontSize={11} fill={contrastText(BORDER_COLOR)} padding={4} listening={false} />
      </Label>

      {/* Processed image — full frame below input */}
      {processedImage && (
        <>
          <KonvaImage
            image={processedImage}
            x={frame.x}
            y={processedY}
            width={frame.width}
            height={frame.height}
            listening={false}
          />
          <Rect
            x={frame.x}
            y={processedY}
            width={frame.width}
            height={frame.height}
            stroke={PROCESSED_BORDER_COLOR}
            strokeWidth={1}
            listening={false}
          />
          <Label x={frame.x} y={processedY - LABEL_HEIGHT} listening={false}>
            <Tag fill={PROCESSED_BORDER_COLOR} cornerRadius={3} />
            <Text text="Processed" fontSize={11} fill={contrastText(PROCESSED_BORDER_COLOR)} padding={4} listening={false} />
          </Label>
        </>
      )}
    </>
  );
}
