import { useMemo, useCallback, useEffect, type ReactNode } from "react";
import { useCanvasStore, type ImageLayer } from "@/stores/canvasStore";
import { useControlStore } from "@/stores/controlStore";
import { UNIT_TYPE_LABELS } from "@/api/types/control";
import { useGenerationStore } from "@/stores/generationStore";
import { useUiStore } from "@/stores/uiStore";
import { useServerInfo } from "@/api/hooks/useServer";
import { ControlUnitControls } from "@/components/generation/tabs/control/ControlUnitControls";
import { LayerPanel } from "@/components/generation/LayerPanel";
import { MaskParams } from "@/components/generation/MaskParams";
import { ChevronUp, ChevronDown, ImagePlus, Trash2, Minimize2, Maximize2, Move, ArrowLeftFromLine, Hand, LocateFixed, Download } from "lucide-react";
import type { FitMode } from "@/lib/image";
import { Button } from "@/components/ui/button";
import { ParamSlider } from "@/components/generation/ParamSlider";
import { contrastText, downloadImage, generateImageFilename, resolveImageSrc } from "@/lib/utils";
import { fileToBase64 } from "@/lib/image";
import { toast } from "sonner";
import { ELEMENT_GAP, PROCESSED_HEADER_HEIGHT, type CanvasLayout, type ControlFramePosition } from "./useControlFrameLayout";

export const HEADER_HEIGHT = 36;
const DRAWER_MAX_HEIGHT = 420;
export const PANEL_WIDTH = 320;
const STROKE_HALF = 1;
const CONTROL_COLOR = "#f59e0b";
export const INPUT_COLOR_ACTIVE = "#4ade80";
export const INPUT_COLOR_REFERENCE = "#38bdf8";
export const INPUT_COLOR_INACTIVE = "#6b7280";
export const OUTPUT_COLOR = "#60a5fa";
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

export interface FrameHeaderProps {
  mode: "panel" | "hat";
  color: string;
  label: string;
  sizeText?: string;
  /** Canvas-space X of the frame's left edge */
  canvasX: number;
  /** Canvas-space Y of the frame's top edge (default 0) */
  canvasY?: number;
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

export function FrameHeader({ mode, color, label, sizeText, canvasX, canvasY = 0, frameW, viewport, labelScale, actions, subHeader, drawer, collapsed, onToggleCollapsed }: FrameHeaderProps) {
  const textColor = contrastText(color);
  const combinedScale = viewport.scale * labelScale;

  const style = useMemo<React.CSSProperties>(() => {
    if (mode === "hat") {
      // Frame-width, flush to top edge, anchor bottom-left
      const anchorX = (canvasX - STROKE_HALF) * viewport.scale + viewport.x;
      const anchorY = (canvasY + STROKE_HALF) * viewport.scale + viewport.y;
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
  }, [mode, canvasX, canvasY, frameW, viewport, labelScale, combinedScale]);

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
  const unifiedIndex = unitIndex + 2;
  const isReference = unit.unitType === "reference";
  const panelColor = isReference ? INPUT_COLOR_REFERENCE : CONTROL_COLOR;
  const textColor = contrastText(panelColor);
  const roleLabel = isReference ? "Reference" : `Control: ${UNIT_TYPE_LABELS[unit.unitType] ?? unit.unitType}`;
  const labelText = `Input ${unifiedIndex} (${roleLabel})`;

  let sizeText: string | null = null;
  if (isOwner) {
    if (isReference) {
      sizeText = imageDims ? `${imageDims.w}\u00d7${imageDims.h}` : `${genW}\u00d7${genH}`;
    } else if (unit.fitMode === "free") {
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

  const subHeader = isOwner && unit.image && !isReference ? (
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
      style={{ borderColor: panelColor }}
      onClick={handlePanelClick}
    >
      <div className="flex flex-col shrink-0 rounded-t-md" style={{ backgroundColor: panelColor }}>
        <div className="flex items-center justify-between px-3" style={{ minHeight: HEADER_HEIGHT }}>
          <span className="text-base font-medium truncate" style={{ color: textColor }}>{labelText}</span>
          <div className="flex items-center gap-0.5 shrink-0">
            {actions}
            {!isReference && (
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

      {!isReference && !collapsed && (
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
  const inputRole = useCanvasStore((s) => s.inputRole);
  const setInputRole = useCanvasStore((s) => s.setInputRole);
  const panelCollapsedOverrides = useCanvasStore((s) => s.panelCollapsedOverrides);
  const togglePanelCollapsed = useCanvasStore((s) => s.togglePanelCollapsed);
  const denoisingStrength = useGenerationStore((s) => s.denoisingStrength);
  const setParam = useGenerationStore((s) => s.setParam);
  const pixelW = useGenerationStore((s) => s.width);
  const pixelH = useGenerationStore((s) => s.height);
  const hasLayers = layers.length > 0;
  const isReference = inputRole === "reference";
  const supportsStrength = useServerInfo().data?.model.supports_strength ?? true;

  // Auto-switch to reference when model doesn't support strength
  useEffect(() => {
    if (!supportsStrength && inputRole === "initial") {
      setInputRole("reference");
    }
  }, [supportsStrength]); // eslint-disable-line react-hooks/exhaustive-deps -- only react to model capability changes

  const handleRoleChange = useCallback((role: "initial" | "reference") => {
    if (role === inputRole) return;
    if (role === "initial" && !supportsStrength) {
      toast.info("This model uses the image as a reference — denoising strength has no effect.");
    }
    setInputRole(role);
  }, [inputRole, setInputRole, supportsStrength]);

  const firstImage = layers.find((l): l is ImageLayer => l.type === "image");

  const override = panelCollapsedOverrides.get(INPUT_PANEL_KEY);
  const collapsed = override !== undefined ? override : !hasLayers;

  const baseSizeText = firstImage ? `${firstImage.naturalWidth}\u00d7${firstImage.naturalHeight}` : `${pixelW}\u00d7${pixelH}`;
  const sizeText = (genSize.width !== pixelW || genSize.height !== pixelH)
    ? `${baseSizeText} \u2192 ${genSize.width}\u00d7${genSize.height}`
    : baseSizeText;

  const handleDenoising = useCallback((v: number) => setParam("denoisingStrength", v), [setParam]);

  const inputColor = !hasLayers ? INPUT_COLOR_INACTIVE : isReference ? INPUT_COLOR_REFERENCE : INPUT_COLOR_ACTIVE;
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

  const roleToggle = (
    <div className="flex items-center gap-0.5 rounded-full p-0.5" style={{ backgroundColor: "rgba(0,0,0,0.15)" }}>
      <button
        onClick={(e) => { e.stopPropagation(); handleRoleChange("initial"); }}
        className="px-2 py-0.5 text-xs font-medium rounded-full transition-colors"
        style={{
          backgroundColor: !isReference ? "rgba(255,255,255,0.25)" : "transparent",
          color: textColor,
        }}
      >
        Initial
      </button>
      <button
        onClick={(e) => { e.stopPropagation(); handleRoleChange("reference"); }}
        className="px-2 py-0.5 text-xs font-medium rounded-full transition-colors"
        style={{
          backgroundColor: isReference ? "rgba(255,255,255,0.25)" : "transparent",
          color: textColor,
        }}
      >
        Reference
      </button>
    </div>
  );

  const drawer = (
    <>
      {!isReference && (
        <ParamSlider label="Denoise" value={denoisingStrength} onChange={handleDenoising} min={0} max={1} step={0.05} disabled={!hasLayers} />
      )}
      <LayerPanel />
      {!isReference && <MaskParams />}
    </>
  );

  const label = isReference ? "Input 1 (Reference)" : "Input 1 (Initial)";

  return (
    <FrameHeader
      mode="panel"
      color={inputColor}
      label={label}
      sizeText={sizeText}
      canvasX={canvasX}
      frameW={frameW}
      viewport={viewport}
      labelScale={labelScale}
      actions={actions}
      subHeader={roleToggle}
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

  const handleDownload = useCallback(() => {
    if (!selectedResult || selectedImageIndex === null) return;
    const raw = selectedResult.images[selectedImageIndex];
    if (!raw) return;
    const src = resolveImageSrc(raw);
    const filename = generateImageFilename(selectedResult.info, selectedImageIndex);
    downloadImage(src, filename);
  }, [selectedResult, selectedImageIndex]);

  const textColor = contrastText(OUTPUT_COLOR);

  const actions = (
    <>
      <Button
        variant="ghost"
        size="icon-xs"
        onClick={handleDownload}
        disabled={!hasSelectedImage}
        title="Download output image"
        className="hover:bg-black/10 disabled:opacity-30"
      >
        <Download size={16} style={{ color: textColor }} />
      </Button>
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
    </>
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

// ─── Processed frame header (uses FrameHeader in hat mode) ──────────────────

function ProcessedFrameHeader({ canvasX, canvasY, viewport, frameW, labelScale, sizeText, label, imageSrc }: {
  canvasX: number;
  canvasY?: number;
  viewport: { x: number; y: number; scale: number };
  frameW: number; labelScale: number; sizeText?: string;
  label?: string;
  imageSrc?: string | null;
}) {
  const compositeProcessed = useControlStore((s) => s.compositeProcessed);
  const units = useControlStore((s) => s.units);

  const processedSrc = useMemo(() => {
    if (imageSrc !== undefined) return imageSrc;
    if (compositeProcessed) return compositeProcessed;
    const first = units.find((u) => u.enabled && !!u.processedImage);
    return first?.processedImage ?? null;
  }, [imageSrc, compositeProcessed, units]);

  const handleDownload = useCallback(() => {
    if (!processedSrc) return;
    const timestamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
    downloadImage(processedSrc, `processed_${timestamp}.png`);
  }, [processedSrc]);

  const textColor = contrastText(PROCESSED_COLOR);

  const actions = (
    <Button
      variant="ghost"
      size="icon-xs"
      onClick={handleDownload}
      disabled={!processedSrc}
      title="Download processed image"
      className="hover:bg-black/10 disabled:opacity-30"
    >
      <Download size={16} style={{ color: textColor }} />
    </Button>
  );

  return (
    <FrameHeader
      mode="hat"
      color={PROCESSED_COLOR}
      label={label ?? "Processed"}
      sizeText={sizeText}
      canvasX={canvasX}
      canvasY={canvasY}
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
  const units = useControlStore((s) => s.units);

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

      {/* Per-unit processed slot headers (below each control frame) */}
      {layout.controlFrames.map((frame) => {
        const activeSlots = frame.processedSlots.filter((s) => {
          const u = units[s.unitIndex];
          return u && !!u.processedImage;
        });
        if (activeSlots.length === 0) return null;
        return activeSlots.map((slot, slotIdx) => {
          const imageY = frame.y + frame.height + ELEMENT_GAP + PROCESSED_HEADER_HEIGHT + slotIdx * (frame.height + ELEMENT_GAP + PROCESSED_HEADER_HEIGHT);
          const slotLabel = activeSlots.length > 1 ? `Processed (Input ${slot.unitIndex + 2})` : "Processed";
          const unit = units[slot.unitIndex];
          return (
            <ProcessedFrameHeader
              key={`proc-${frame.unitIndex}-${slot.unitIndex}`}
              canvasX={frame.x}
              canvasY={imageY}
              viewport={viewport}
              frameW={frame.width}
              labelScale={labelScale}
              label={slotLabel}
              imageSrc={unit?.processedImage ?? null}
            />
          );
        });
      })}

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
        <ProcessedFrameHeader
          canvasX={layout.processedX}
          viewport={viewport}
          frameW={displayW}
          labelScale={labelScale}
          sizeText={genSizeText}
        />
      )}
    </>
  );
}
