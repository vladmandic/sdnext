import { useEffect, useMemo, useState } from "react";
import { useCanvasStore } from "@/stores/canvasStore";
import { useControlStore } from "@/stores/controlStore";
import { useGenerationStore } from "@/stores/generationStore";
import { useUiStore } from "@/stores/uiStore";
import { ControlUnitControls } from "@/components/generation/tabs/control/ControlUnitControls";
import { ChevronUp, ChevronDown, ImagePlus, Trash2, Minimize2, Maximize2, Move } from "lucide-react";
import type { FitMode } from "@/lib/image";
import { Button } from "@/components/ui/button";
import { contrastText } from "@/lib/utils";
import { ELEMENT_GAP, type CanvasLayout } from "./useControlFrameLayout";

function useImageDimensions(file: File | null): { w: number; h: number } | null {
  const [dims, setDims] = useState<{ w: number; h: number } | null>(null);
  useEffect(() => {
    if (!file) return;
    const url = URL.createObjectURL(file);
    const img = new window.Image();
    img.onload = () => { setDims({ w: img.naturalWidth, h: img.naturalHeight }); URL.revokeObjectURL(url); };
    img.onerror = () => URL.revokeObjectURL(url);
    img.src = url;
    return () => URL.revokeObjectURL(url);
  }, [file]);
  // Clear dims synchronously when file is removed (not inside effect)
  if (!file && dims) return null;
  return dims;
}

const HEADER_HEIGHT = 36;
const DRAWER_MAX_HEIGHT = 420;
const PANEL_WIDTH = 320;
const STROKE_HALF = 1; // Konva strokeWidth=2 → 1 canvas unit extends outside each edge
const CONTROL_COLOR = "#f59e0b";
const INPUT_COLOR = "#4ade80";
const OUTPUT_COLOR = "#60a5fa";

interface ControlFramePanelProps {
  layout: CanvasLayout;
  unitIndex: number;
  collapsed: boolean;
  onPickImage?: (unitIndex: number) => void;
  onClearImage?: (unitIndex: number) => void;
}

function ControlFramePanel({ layout, unitIndex, collapsed, onPickImage, onClearImage }: ControlFramePanelProps) {
  const togglePanelCollapsed = useCanvasStore((s) => s.togglePanelCollapsed);
  const setSelectedControlFrame = useCanvasStore((s) => s.setSelectedControlFrame);
  const viewport = useCanvasStore((s) => s.viewport);
  const units = useControlStore((s) => s.units);
  const setUnitParam = useControlStore((s) => s.setUnitParam);
  const labelScale = useUiStore((s) => s.canvasLabelScale);

  const style = useMemo(() => {
    const frame = layout.controlFrames.find((f) => f.unitIndex === unitIndex);
    if (!frame) return null;

    // Center above the frame with a gap
    const screenCenterX = (frame.x + frame.width / 2) * viewport.scale + viewport.x;
    const screenTopY = frame.y * viewport.scale + viewport.y - ELEMENT_GAP * viewport.scale;
    const combinedScale = viewport.scale * labelScale;

    return {
      position: "absolute" as const,
      left: `${screenCenterX - PANEL_WIDTH / 2}px`,
      bottom: `calc(100% - ${screenTopY}px)`,
      width: `${PANEL_WIDTH}px`,
      borderColor: CONTROL_COLOR,
      transform: `scale(${combinedScale})`,
      transformOrigin: "bottom center",
    };
  }, [unitIndex, layout.controlFrames, viewport, labelScale]);

  const unit = units[unitIndex];
  const imageDims = useImageDimensions(unit?.image ?? null);

  if (!style) return null;
  if (!unit) return null;

  const frame = layout.controlFrames.find((f) => f.unitIndex === unitIndex);
  const fitSuffix = unit.fitMode === "contain" ? "fit" : unit.fitMode === "cover" ? "crop" : "stretch";
  const sizeText = imageDims ? `${imageDims.w}×${imageDims.h} ${fitSuffix}` : frame ? `${frame.width}×${frame.height}` : null;
  const labelText = `Control ${unitIndex} (${unit.unitType})`;
  const textColor = contrastText(CONTROL_COLOR);

  const handlePanelClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    setSelectedControlFrame(unitIndex);
  };

  return (
    <div
      style={style}
      className="flex flex-col overflow-hidden rounded-t-md border shadow-lg z-50"
      onClick={handlePanelClick}
    >
      {/* Header bar — always at top, acts as drawer front */}
      <div
        className="flex items-center justify-between px-3 shrink-0 rounded-t-md"
        style={{ backgroundColor: CONTROL_COLOR, height: HEADER_HEIGHT }}
      >
        <span className="text-base font-medium truncate" style={{ color: textColor }}>{labelText}</span>
        {sizeText && <span className="text-xs opacity-70 shrink-0" style={{ color: textColor }}>{sizeText}</span>}
        <div className="flex items-center gap-0.5 shrink-0">
          {unit.image && (
            <>
              <Button
                variant="ghost"
                size="icon-xs"
                onClick={(e) => {
                  e.stopPropagation();
                  const next: FitMode = unit.fitMode === "contain" ? "cover" : unit.fitMode === "cover" ? "fill" : "contain";
                  setUnitParam(unitIndex, "fitMode", next);
                }}
                title={`Fit: ${unit.fitMode}`}
                className="hover:bg-black/10"
              >
                {unit.fitMode === "contain" ? <Minimize2 size={16} style={{ color: textColor }} /> : unit.fitMode === "cover" ? <Maximize2 size={16} style={{ color: textColor }} /> : <Move size={16} style={{ color: textColor }} />}
              </Button>
              <Button
                variant="ghost"
                size="icon-xs"
                onClick={(e) => {
                  e.stopPropagation();
                  onPickImage?.(unitIndex);
                }}
                title="Replace image"
                className="hover:bg-black/10"
              >
                <ImagePlus size={16} style={{ color: textColor }} />
              </Button>
              <Button
                variant="ghost"
                size="icon-xs"
                onClick={(e) => {
                  e.stopPropagation();
                  onClearImage?.(unitIndex);
                }}
                title="Clear image"
                className="hover:bg-black/10"
              >
                <Trash2 size={16} style={{ color: textColor }} />
              </Button>
            </>
          )}
          <Button
            variant="ghost"
            size="icon-xs"
            onClick={(e) => {
              e.stopPropagation();
              togglePanelCollapsed(unitIndex, collapsed);
            }}
            title={collapsed ? "Expand settings" : "Collapse settings"}
            className="hover:bg-black/10"
          >
            {collapsed ? <ChevronDown size={16} style={{ color: textColor }} /> : <ChevronUp size={16} style={{ color: textColor }} />}
          </Button>
        </div>
      </div>

      {/* Controls drawer — hangs below header, panel grows upward */}
      {!collapsed && (
        <div
          className="p-3 overflow-y-auto bg-background/95 backdrop-blur-sm border-t border-border/50"
          style={{ maxHeight: DRAWER_MAX_HEIGHT }}
        >
          <ControlUnitControls index={unitIndex} compact />
        </div>
      )}
    </div>
  );
}

/** Container that renders header bars for all frames */
interface ControlFramePanelsProps {
  layout: CanvasLayout;
  onPickImage?: (unitIndex: number) => void;
  onClearImage?: (unitIndex: number) => void;
}

export function ControlFramePanels({ layout, onPickImage, onClearImage }: ControlFramePanelsProps) {
  const panelCollapsedOverrides = useCanvasStore((s) => s.panelCollapsedOverrides);
  const units = useControlStore((s) => s.units);
  const viewport = useCanvasStore((s) => s.viewport);
  const frameW = useGenerationStore((s) => s.width);
  const frameH = useGenerationStore((s) => s.height);
  const labelScale = useUiStore((s) => s.canvasLabelScale);

  return (
    <>
      {/* Control frame panels */}
      {layout.controlFrames.length > 0 && units.map((unit, index) => {
        const hasImage = unit.image !== null;
        const override = panelCollapsedOverrides.get(index);
        const isCollapsed = override !== undefined ? override : !hasImage;

        return (
          <ControlFramePanel
            key={index}
            unitIndex={index}
            collapsed={isCollapsed}
            layout={layout}
            onPickImage={onPickImage}
            onClearImage={onClearImage}
          />
        );
      })}

      {/* Input frame header bar — img2img only */}
      {layout.showInputFrame && (
        <FrameHeaderBar label="Input" color={INPUT_COLOR} canvasX={layout.inputX} viewport={viewport} frameW={frameW} frameH={frameH} labelScale={labelScale} />
      )}

      {/* Output frame header bar — always present */}
      <FrameHeaderBar label="Output" color={OUTPUT_COLOR} canvasX={layout.outputX} viewport={viewport} frameW={frameW} frameH={frameH} labelScale={labelScale} />
    </>
  );
}

interface FrameHeaderBarProps {
  label: string;
  color: string;
  canvasX: number;
  frameW: number;
  frameH: number;
  viewport: { x: number; y: number; scale: number };
  labelScale: number;
}

function FrameHeaderBar({ label, color, canvasX, frameW, frameH, viewport, labelScale }: FrameHeaderBarProps) {
  // Anchor bottom-left to frame's top-left, overlap top stroke so header acts as top border
  const anchorX = (canvasX - STROKE_HALF) * viewport.scale + viewport.x;
  const anchorY = (0 + STROKE_HALF) * viewport.scale + viewport.y;
  // Width must match the frame exactly — divide by labelScale to counteract the uniform scale
  const canvasWidth = (frameW + STROKE_HALF * 2) / labelScale;
  const textColor = contrastText(color);
  const combinedScale = viewport.scale * labelScale;

  const style: React.CSSProperties = {
    position: "absolute",
    left: `${anchorX}px`,
    bottom: `calc(100% - ${anchorY}px)`,
    width: `${canvasWidth}px`,
    height: HEADER_HEIGHT,
    backgroundColor: color,
    transform: `scale(${combinedScale})`,
    transformOrigin: "bottom left",
  };

  return (
    <div style={style} className="flex items-center justify-between px-3 rounded-t-md z-50">
      <span className="text-base font-medium" style={{ color: textColor }}>{label}</span>
      <span className="text-xs opacity-70" style={{ color: textColor }}>{frameW}×{frameH}</span>
    </div>
  );
}
