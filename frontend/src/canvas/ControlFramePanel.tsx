import { useMemo, useCallback, type ReactNode } from "react";
import { useCanvasStore, type ImageLayer } from "@/stores/canvasStore";
import { useControlStore } from "@/stores/controlStore";
import { useGenerationStore } from "@/stores/generationStore";
import { useUiStore } from "@/stores/uiStore";
import { ControlUnitControls } from "@/components/generation/tabs/control/ControlUnitControls";
import { LayerPanel } from "@/components/generation/LayerPanel";
import { MaskParams } from "@/components/generation/MaskParams";
import { ChevronUp, ChevronDown, ImagePlus, Trash2, Minimize2, Maximize2, Move, ArrowLeftFromLine, Hand, LocateFixed } from "lucide-react";
import type { FitMode } from "@/lib/image";
import { Button } from "@/components/ui/button";
import { ParamSlider } from "@/components/generation/ParamSlider";
import { contrastText } from "@/lib/utils";
import { fileToBase64 } from "@/lib/image";
import { ELEMENT_GAP, type CanvasLayout, type ControlFramePosition } from "./useControlFrameLayout";

const HEADER_HEIGHT = 36;
const DRAWER_MAX_HEIGHT = 420;
const PANEL_WIDTH = 320;
const STROKE_HALF = 1;
const CONTROL_COLOR = "#f59e0b";
const INPUT_COLOR_ACTIVE = "#4ade80";
const INPUT_COLOR_INACTIVE = "#6b7280";
const OUTPUT_COLOR = "#60a5fa";
const PROCESSED_COLOR = "#c084fc";
const INPUT_PANEL_KEY = -1;

// ─── Unified FrameHeader ────────────────────────────────────────────────────
//
// Two modes:
//   "panel" — fixed 320px width, centered above frame with gap, expandable drawer
//   "hat"   — matches frame width, flush to frame top, no drawer
//
// Visual affordance: hats are flat bars = nothing to expand.
//                    Panels have rounded top + border = expandable.

interface FrameHeaderProps {
  mode: "panel" | "hat";
  color: string;
  label: string;
  sizeText?: string;
  /** Canvas-space X of the frame's left edge */
  canvasX: number;
  /** Canvas-space frame width */
  frameW: number;
  viewport: { x: number; y: number; scale: number };
  labelScale: number;
  /** Buttons rendered to the right of the label (inside the colored header) */
  actions?: ReactNode;
  /** Extra content between the label row and the actions (e.g. size+fit row) */
  subHeader?: ReactNode;
  /** Drawer content (panel mode only) — rendered below header when expanded */
  drawer?: ReactNode;
  /** Whether the drawer is collapsed (panel mode only) */
  collapsed?: boolean;
  /** Toggle callback for the chevron (panel mode only) */
  onToggleCollapsed?: () => void;
}

function FrameHeader({ mode, color, label, sizeText, canvasX, frameW, viewport, labelScale, actions, subHeader, drawer, collapsed, onToggleCollapsed }: FrameHeaderProps) {
  const textColor = contrastText(color);
  const combinedScale = viewport.scale * labelScale;

  const style = useMemo<React.CSSProperties>(() => {
    if (mode === "hat") {
      // Frame-width, flush to top edge, anchor bottom-left
      const anchorX = (canvasX - STROKE_HALF) * viewport.scale + viewport.x;
      const anchorY = STROKE_HALF * viewport.scale + viewport.y;
      const widthPx = (frameW + STROKE_HALF * 2) / labelScale;
      return {
        position: "absolute",
        left: `${anchorX}px`,
        bottom: `calc(100% - ${anchorY}px)`,
        width: `${widthPx}px`,
        transform: `scale(${combinedScale})`,
        transformOrigin: "bottom left",
      };
    }
    // Panel mode: fixed width, positioned above frame with gap.
    // Left-align to frame edge so the panel never overlaps adjacent frames.
    const screenLeftX = (canvasX - STROKE_HALF) * viewport.scale + viewport.x;
    const screenTopY = STROKE_HALF * viewport.scale + viewport.y - ELEMENT_GAP * viewport.scale;
    return {
      position: "absolute",
      left: `${screenLeftX}px`,
      bottom: `calc(100% - ${screenTopY}px)`,
      width: `${PANEL_WIDTH}px`,
      transform: `scale(${combinedScale})`,
      transformOrigin: "bottom left",
    };
  }, [mode, canvasX, frameW, viewport, labelScale, combinedScale]);

  const isPanel = mode === "panel";
  const showDrawer = isPanel && drawer && !collapsed;
  const showChevron = isPanel && drawer !== undefined && onToggleCollapsed;

  return (
    <div style={style} className="z-50">
      <div
        className={`flex flex-col overflow-hidden ${isPanel ? "rounded-t-md border shadow-lg" : ""}`}
        style={isPanel ? { borderColor: color } : undefined}
      >
        {/* Colored header */}
        <div
          className={`flex flex-col shrink-0 ${isPanel ? "rounded-t-md" : "rounded-t-md"}`}
          style={{ backgroundColor: color, minHeight: HEADER_HEIGHT }}
        >
          {/* Row 1: label + actions + chevron */}
          <div className="flex items-center justify-between px-3" style={{ minHeight: HEADER_HEIGHT }}>
            <span className="text-base font-medium truncate" style={{ color: textColor }}>{label}</span>
            <div className="flex items-center gap-0.5 shrink-0">
              {actions}
              {showChevron && (
                <Button
                  variant="ghost"
                  size="icon-xs"
                  onClick={(e) => { e.stopPropagation(); onToggleCollapsed!(); }}
                  title={collapsed ? "Expand settings" : "Collapse settings"}
                  className="hover:bg-black/10"
                >
                  {collapsed ? <ChevronDown size={16} style={{ color: textColor }} /> : <ChevronUp size={16} style={{ color: textColor }} />}
                </Button>
              )}
              {/* Size text on the right for hats (no chevron) */}
              {!isPanel && sizeText && (
                <span className="text-xs opacity-70 ml-1" style={{ color: textColor }}>{sizeText}</span>
              )}
            </div>
          </div>
          {/* Sub-header row (size text for panels, or custom content) */}
          {isPanel && (sizeText || subHeader) && (
            <div className="flex items-center justify-between px-3 pb-1.5">
              {sizeText && <span className="text-xs opacity-70" style={{ color: textColor }}>{sizeText}</span>}
              {subHeader}
            </div>
          )}
        </div>

        {/* Drawer (panel mode only) */}
        {showDrawer && (
          <div
            className="p-3 overflow-y-auto bg-background/95 backdrop-blur-sm border-t border-border/50 flex flex-col gap-2"
            style={{ maxHeight: DRAWER_MAX_HEIGHT }}
          >
            {drawer}
          </div>
        )}
      </div>
    </div>
  );
}

// ─── Control unit panel (one per unit, no positioning) ──────────────────────

interface UnitPanelProps {
  unitIndex: number;
  isOwner: boolean;
  collapsed: boolean;
  genSize: { width: number; height: number };
  onPickImage?: (unitIndex: number) => void;
  onClearImage?: (unitIndex: number) => void;
}

function UnitPanel({ unitIndex, isOwner, collapsed, genSize, onPickImage, onClearImage }: UnitPanelProps) {
  const togglePanelCollapsed = useCanvasStore((s) => s.togglePanelCollapsed);
  const setSelectedControlFrame = useCanvasStore((s) => s.setSelectedControlFrame);
  const unit = useControlStore((s) => s.units[unitIndex]);
  const setUnitParam = useControlStore((s) => s.setUnitParam);
  const setFreeTransform = useControlStore((s) => s.setFreeTransform);
  const { width: genW, height: genH } = genSize;

  if (!unit) return null;

  const imageDims = isOwner ? unit.imageDims : null;
  const textColor = contrastText(CONTROL_COLOR);
  const labelText = `Unit ${unitIndex} (${unit.unitType})`;
  const isAsset = unit.unitType === "asset";

  let sizeText: string | null = null;
  if (isOwner) {
    if (unit.fitMode === "free") {
      sizeText = imageDims ? `${imageDims.w}\u00d7${imageDims.h} free` : `${genW}\u00d7${genH}`;
    } else {
      const fitSuffix = unit.fitMode === "contain" ? "fit" : unit.fitMode === "cover" ? "crop" : "stretch";
      sizeText = imageDims ? `${imageDims.w}\u00d7${imageDims.h} \u2192 ${genW}\u00d7${genH} ${fitSuffix}` : `${genW}\u00d7${genH}`;
    }
  }

  const handlePanelClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    setSelectedControlFrame(unitIndex);
  };

  const actions = (
    <>
      {isOwner && unit.image && (
        <>
          <Button variant="ghost" size="icon-xs" onClick={(e) => { e.stopPropagation(); onPickImage?.(unitIndex); }} title="Replace image" className="hover:bg-black/10">
            <ImagePlus size={16} style={{ color: textColor }} />
          </Button>
          <Button variant="ghost" size="icon-xs" onClick={(e) => { e.stopPropagation(); onClearImage?.(unitIndex); }} title="Clear image" className="hover:bg-black/10">
            <Trash2 size={16} style={{ color: textColor }} />
          </Button>
        </>
      )}
    </>
  );

  const fitIcon = unit.fitMode === "contain" ? <Minimize2 size={16} style={{ color: textColor }} />
    : unit.fitMode === "cover" ? <Maximize2 size={16} style={{ color: textColor }} />
    : unit.fitMode === "fill" ? <Move size={16} style={{ color: textColor }} />
    : <Hand size={16} style={{ color: textColor }} />;

  const subHeader = isOwner && unit.image ? (
    <div className="flex items-center gap-0.5">
      <Button
        variant="ghost"
        size="icon-xs"
        onClick={(e) => {
          e.stopPropagation();
          const next: FitMode = unit.fitMode === "contain" ? "cover" : unit.fitMode === "cover" ? "fill" : unit.fitMode === "fill" ? "free" : "contain";
          if (next === "free" || unit.fitMode === "free") setFreeTransform(unitIndex, null);
          setUnitParam(unitIndex, "fitMode", next);
        }}
        title={`Fit: ${unit.fitMode}`}
        className="hover:bg-black/10"
      >
        {fitIcon}
      </Button>
      {unit.fitMode === "free" && unit.freeTransform !== null && (
        <Button
          variant="ghost"
          size="icon-xs"
          onClick={(e) => { e.stopPropagation(); setFreeTransform(unitIndex, null); }}
          title="Re-center image"
          className="hover:bg-black/10"
        >
          <LocateFixed size={16} style={{ color: textColor }} />
        </Button>
      )}
    </div>
  ) : undefined;

  // UnitPanel is rendered inside a positioned container (ControlFrameStack),
  // so it doesn't use FrameHeader's positioning — it renders the visual parts directly.
  return (
    <div
      className="flex flex-col overflow-hidden rounded-t-md border shadow-lg"
      style={{ borderColor: CONTROL_COLOR }}
      onClick={handlePanelClick}
    >
      <div className="flex flex-col shrink-0 rounded-t-md" style={{ backgroundColor: CONTROL_COLOR }}>
        <div className="flex items-center justify-between px-3" style={{ minHeight: HEADER_HEIGHT }}>
          <span className="text-base font-medium truncate" style={{ color: textColor }}>{labelText}</span>
          <div className="flex items-center gap-0.5 shrink-0">
            {actions}
            {!isAsset && (
              <Button
                variant="ghost"
                size="icon-xs"
                onClick={(e) => { e.stopPropagation(); togglePanelCollapsed(unitIndex, collapsed); }}
                title={collapsed ? "Expand settings" : "Collapse settings"}
                className="hover:bg-black/10"
              >
                {collapsed ? <ChevronDown size={16} style={{ color: textColor }} /> : <ChevronUp size={16} style={{ color: textColor }} />}
              </Button>
            )}
          </div>
        </div>
        {sizeText && (
          <div className="flex items-center justify-between px-3 pb-1.5">
            <span className="text-xs opacity-70" style={{ color: textColor }}>{sizeText}</span>
            {subHeader}
          </div>
        )}
      </div>

      {!isAsset && !collapsed && (
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

// ─── ControlFrameStack: positioned container for stacked unit panels ────────

interface ControlFrameStackProps {
  frame: ControlFramePosition;
  genSize: { width: number; height: number };
  onPickImage?: (unitIndex: number) => void;
  onClearImage?: (unitIndex: number) => void;
}

function ControlFrameStack({ frame, genSize, onPickImage, onClearImage }: ControlFrameStackProps) {
  const viewport = useCanvasStore((s) => s.viewport);
  const labelScale = useUiStore((s) => s.canvasLabelScale);
  const panelCollapsedOverrides = useCanvasStore((s) => s.panelCollapsedOverrides);
  const units = useControlStore((s) => s.units);

  const containerStyle = useMemo(() => {
    const screenLeftX = (frame.x - STROKE_HALF) * viewport.scale + viewport.x;
    const screenTopY = frame.y * viewport.scale + viewport.y - ELEMENT_GAP * viewport.scale;
    const combinedScale = viewport.scale * labelScale;

    return {
      position: "absolute" as const,
      left: `${screenLeftX}px`,
      bottom: `calc(100% - ${screenTopY}px)`,
      width: `${PANEL_WIDTH}px`,
      transform: `scale(${combinedScale})`,
      transformOrigin: "bottom left",
      display: "flex",
      flexDirection: "column" as const,
      gap: "4px",
    };
  }, [frame, viewport, labelScale]);

  const referencingSlots = frame.processedSlots.filter((s) => s.unitIndex !== frame.unitIndex);

  const ownerUnit = units[frame.unitIndex];
  if (!ownerUnit) return null;

  const ownerHasImage = ownerUnit.image !== null;
  const ownerOverride = panelCollapsedOverrides.get(frame.unitIndex);
  const ownerCollapsed = ownerOverride !== undefined ? ownerOverride : !ownerHasImage;

  return (
    <div style={containerStyle} className="z-50">
      {referencingSlots.map((slot) => {
        const override = panelCollapsedOverrides.get(slot.unitIndex);
        const isCollapsed = override !== undefined ? override : true;
        return (
          <UnitPanel
            key={slot.unitIndex}
            unitIndex={slot.unitIndex}
            isOwner={false}
            collapsed={isCollapsed}
            genSize={genSize}
          />
        );
      })}
      <UnitPanel
        unitIndex={frame.unitIndex}
        isOwner
        collapsed={ownerCollapsed}
        genSize={genSize}
        onPickImage={onPickImage}
        onClearImage={onClearImage}
      />
    </div>
  );
}

// ─── Input frame panel (uses FrameHeader in panel mode) ─────────────────────

function InputFramePanel({ canvasX, frameW, genSize, viewport, labelScale, onPickImage, onClearAll }: {
  canvasX: number; frameW: number;
  genSize: { width: number; height: number };
  viewport: { x: number; y: number; scale: number };
  labelScale: number;
  onPickImage?: () => void;
  onClearAll?: () => void;
}) {
  const layers = useCanvasStore((s) => s.layers);
  const panelCollapsedOverrides = useCanvasStore((s) => s.panelCollapsedOverrides);
  const togglePanelCollapsed = useCanvasStore((s) => s.togglePanelCollapsed);
  const denoisingStrength = useGenerationStore((s) => s.denoisingStrength);
  const setParam = useGenerationStore((s) => s.setParam);
  const pixelW = useGenerationStore((s) => s.width);
  const pixelH = useGenerationStore((s) => s.height);
  const hasLayers = layers.length > 0;

  const firstImage = layers.find((l): l is ImageLayer => l.type === "image");

  const override = panelCollapsedOverrides.get(INPUT_PANEL_KEY);
  const collapsed = override !== undefined ? override : !hasLayers;

  const baseSizeText = firstImage ? `${firstImage.naturalWidth}\u00d7${firstImage.naturalHeight}` : `${pixelW}\u00d7${pixelH}`;
  const sizeText = (genSize.width !== pixelW || genSize.height !== pixelH)
    ? `${baseSizeText} \u2192 ${genSize.width}\u00d7${genSize.height}`
    : baseSizeText;

  const handleDenoising = useCallback((v: number) => setParam("denoisingStrength", v), [setParam]);

  const inputColor = hasLayers ? INPUT_COLOR_ACTIVE : INPUT_COLOR_INACTIVE;
  const textColor = contrastText(inputColor);

  const actions = hasLayers ? (
    <>
      <Button variant="ghost" size="icon-xs" onClick={() => onPickImage?.()} title="Add image" className="hover:bg-black/10">
        <ImagePlus size={16} style={{ color: textColor }} />
      </Button>
      <Button variant="ghost" size="icon-xs" onClick={() => onClearAll?.()} title="Clear all" className="hover:bg-black/10">
        <Trash2 size={16} style={{ color: textColor }} />
      </Button>
    </>
  ) : undefined;

  const drawer = (
    <>
      <ParamSlider label="Denoise" value={denoisingStrength} onChange={handleDenoising} min={0} max={1} step={0.05} disabled={!hasLayers} />
      <LayerPanel />
      <MaskParams />
    </>
  );

  return (
    <FrameHeader
      mode="panel"
      color={inputColor}
      label="Input"
      sizeText={sizeText}
      canvasX={canvasX}
      frameW={frameW}
      viewport={viewport}
      labelScale={labelScale}
      actions={actions}
      drawer={drawer}
      collapsed={collapsed}
      onToggleCollapsed={() => togglePanelCollapsed(INPUT_PANEL_KEY, collapsed)}
    />
  );
}

// ─── Output frame header (uses FrameHeader in hat mode) ─────────────────────

function OutputFrameHeader({ canvasX, viewport, frameW, labelScale, sizeText }: {
  canvasX: number;
  viewport: { x: number; y: number; scale: number };
  frameW: number; labelScale: number; sizeText: string;
}) {
  const selectedResultId = useGenerationStore((s) => s.selectedResultId);
  const selectedImageIndex = useGenerationStore((s) => s.selectedImageIndex);
  const results = useGenerationStore((s) => s.results);
  const addImageLayer = useCanvasStore((s) => s.addImageLayer);

  const selectedResult = useMemo(
    () => results.find((r) => r.id === selectedResultId),
    [results, selectedResultId],
  );

  const hasSelectedImage = selectedResult !== undefined && selectedImageIndex !== null && selectedResult.images[selectedImageIndex] !== undefined;

  const handleSendToInput = useCallback(async () => {
    if (!selectedResult || selectedImageIndex === null) return;
    const imageUrl = selectedResult.images[selectedImageIndex];
    if (!imageUrl) return;
    const resp = await fetch(imageUrl);
    const blob = await resp.blob();
    const file = new File([blob], "from-output.png", { type: "image/png" });
    const base64 = await fileToBase64(file);
    const objectUrl = URL.createObjectURL(file);
    const img = new window.Image();
    img.src = objectUrl;
    await new Promise<void>((r) => { img.onload = () => r(); });
    addImageLayer(file, base64, objectUrl, img.naturalWidth, img.naturalHeight);
  }, [selectedResult, selectedImageIndex, addImageLayer]);

  const textColor = contrastText(OUTPUT_COLOR);

  const actions = (
    <Button
      variant="ghost"
      size="icon-xs"
      onClick={handleSendToInput}
      disabled={!hasSelectedImage}
      title="Send selected output to Input frame"
      className="hover:bg-black/10 disabled:opacity-30"
    >
      <ArrowLeftFromLine size={16} style={{ color: textColor }} />
    </Button>
  );

  return (
    <FrameHeader
      mode="hat"
      color={OUTPUT_COLOR}
      label="Output"
      sizeText={sizeText}
      canvasX={canvasX}
      frameW={frameW}
      viewport={viewport}
      labelScale={labelScale}
      actions={actions}
    />
  );
}

// ─── Top-level: renders all frame panels and headers ────────────────────────

interface ControlFramePanelsProps {
  layout: CanvasLayout;
  onPickImage?: (unitIndex: number) => void;
  onClearImage?: (unitIndex: number) => void;
  onClearAll?: () => void;
}

export function ControlFramePanels({ layout, onPickImage, onClearImage, onClearAll }: ControlFramePanelsProps) {
  const viewport = useCanvasStore((s) => s.viewport);
  const labelScale = useUiStore((s) => s.canvasLabelScale);

  const { genSize, displayW } = layout;
  const genSizeText = `${genSize.width}\u00d7${genSize.height}`;

  return (
    <>
      {layout.controlFrames.map((frame) => (
        <ControlFrameStack
          key={frame.unitIndex}
          frame={frame}
          genSize={genSize}
          onPickImage={onPickImage}
          onClearImage={onClearImage}
        />
      ))}

      <InputFramePanel
        canvasX={layout.inputX}
        frameW={displayW}
        genSize={genSize}
        viewport={viewport}
        labelScale={labelScale}
        onPickImage={() => onPickImage?.(-1)}
        onClearAll={onClearAll}
      />

      <OutputFrameHeader
        canvasX={layout.outputX}
        viewport={viewport}
        frameW={displayW}
        labelScale={labelScale}
        sizeText={genSizeText}
      />

      {layout.showProcessedFrame && (
        <FrameHeader
          mode="hat"
          color={PROCESSED_COLOR}
          label="Processed"
          sizeText={genSizeText}
          canvasX={layout.processedX}
          frameW={displayW}
          viewport={viewport}
          labelScale={labelScale}
        />
      )}
    </>
  );
}
