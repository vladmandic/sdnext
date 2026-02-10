import { useEffect, useMemo } from "react";
import { useCanvasStore } from "@/stores/canvasStore";
import { useControlStore } from "@/stores/controlStore";
import { ControlUnitControls } from "@/components/generation/tabs/control/ControlUnitControls";
import { X } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { CanvasLayout } from "./useControlFrameLayout";

const PANEL_WIDTH = 280;
const PANEL_MAX_HEIGHT = 400;
const PANEL_GAP = 8;

interface ControlFramePanelProps {
  layout: CanvasLayout;
}

export function ControlFramePanel({ layout }: ControlFramePanelProps) {
  const selectedControlFrame = useCanvasStore((s) => s.selectedControlFrame);
  const setSelectedControlFrame = useCanvasStore((s) => s.setSelectedControlFrame);
  const viewport = useCanvasStore((s) => s.viewport);
  const units = useControlStore((s) => s.units);

  // Escape key dismisses panel
  useEffect(() => {
    if (selectedControlFrame === null) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setSelectedControlFrame(null);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [selectedControlFrame, setSelectedControlFrame]);

  // Compute screen position from canvas frame position + viewport
  // Panel sits above the frame, anchored by its bottom edge
  const style = useMemo(() => {
    if (selectedControlFrame === null) return null;
    const frame = layout.controlFrames.find((f) => f.unitIndex === selectedControlFrame);
    if (!frame) return null;

    const screenCenterX = (frame.x + frame.width / 2) * viewport.scale + viewport.x;
    const screenTopY = frame.y * viewport.scale + viewport.y - PANEL_GAP;

    return {
      position: "absolute" as const,
      left: `${screenCenterX - PANEL_WIDTH / 2}px`,
      bottom: `calc(100% - ${screenTopY}px)`,
      width: `${PANEL_WIDTH}px`,
      maxHeight: `${PANEL_MAX_HEIGHT}px`,
    };
  }, [selectedControlFrame, layout.controlFrames, viewport]);

  if (selectedControlFrame === null || !style) return null;

  const unit = units[selectedControlFrame];
  if (!unit) return null;

  const labelText = `Control ${selectedControlFrame} (${unit.unitType})`;

  return (
    <div
      style={style}
      className="overflow-y-auto rounded-lg border border-border bg-background/95 backdrop-blur-sm shadow-lg p-3 z-50"
      onClick={(e) => e.stopPropagation()}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-medium text-foreground">{labelText}</span>
        <Button
          variant="ghost"
          size="icon-xs"
          onClick={() => setSelectedControlFrame(null)}
          title="Close panel"
        >
          <X size={12} />
        </Button>
      </div>

      {/* Controls */}
      <ControlUnitControls index={selectedControlFrame} compact />
    </div>
  );
}
