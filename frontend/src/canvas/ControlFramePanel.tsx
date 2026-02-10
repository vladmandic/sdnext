import { useMemo } from "react";
import { useCanvasStore } from "@/stores/canvasStore";
import { useControlStore } from "@/stores/controlStore";
import { ControlUnitControls } from "@/components/generation/tabs/control/ControlUnitControls";
import { ChevronUp, ChevronDown } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { CanvasLayout } from "./useControlFrameLayout";

const PANEL_WIDTH = 280;
const PANEL_MAX_HEIGHT = 400;
const PANEL_GAP = 8;
const COLLAPSED_HEIGHT = 28;

interface ControlFramePanelProps {
  layout: CanvasLayout;
  unitIndex: number;
  collapsed: boolean;
}

function ControlFramePanel({ layout, unitIndex, collapsed }: ControlFramePanelProps) {
  const togglePanelCollapsed = useCanvasStore((s) => s.togglePanelCollapsed);
  const setSelectedControlFrame = useCanvasStore((s) => s.setSelectedControlFrame);
  const viewport = useCanvasStore((s) => s.viewport);
  const units = useControlStore((s) => s.units);

  // Compute screen position from canvas frame position + viewport
  const style = useMemo(() => {
    const frame = layout.controlFrames.find((f) => f.unitIndex === unitIndex);
    if (!frame) return null;

    const screenCenterX = (frame.x + frame.width / 2) * viewport.scale + viewport.x;
    const screenTopY = frame.y * viewport.scale + viewport.y - PANEL_GAP;

    if (collapsed) {
      return {
        position: "absolute" as const,
        left: `${screenCenterX - PANEL_WIDTH / 2}px`,
        bottom: `calc(100% - ${screenTopY}px)`,
        width: `${PANEL_WIDTH}px`,
        height: `${COLLAPSED_HEIGHT}px`,
      };
    }

    return {
      position: "absolute" as const,
      left: `${screenCenterX - PANEL_WIDTH / 2}px`,
      bottom: `calc(100% - ${screenTopY}px)`,
      width: `${PANEL_WIDTH}px`,
      maxHeight: `${PANEL_MAX_HEIGHT}px`,
    };
  }, [unitIndex, layout.controlFrames, viewport, collapsed]);

  if (!style) return null;

  const unit = units[unitIndex];
  if (!unit) return null;

  const labelText = `Control ${unitIndex} (${unit.unitType})`;

  const handlePanelClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    setSelectedControlFrame(unitIndex);
  };

  return (
    <div
      style={style}
      className="overflow-hidden rounded-lg border border-border bg-background/95 backdrop-blur-sm shadow-lg z-50"
      onClick={handlePanelClick}
    >
      {/* Header - always visible */}
      <div className="flex items-center justify-between px-3 py-1.5 border-b border-border/50 bg-muted/30">
        <span className="text-xs font-medium text-foreground truncate">{labelText}</span>
        <Button
          variant="ghost"
          size="icon-xs"
          onClick={(e) => {
            e.stopPropagation();
            togglePanelCollapsed(unitIndex, collapsed);
          }}
          title={collapsed ? "Expand panel" : "Collapse panel"}
        >
          {collapsed ? <ChevronDown size={12} /> : <ChevronUp size={12} />}
        </Button>
      </div>

      {/* Controls - only when expanded */}
      {!collapsed && (
        <div className="p-3 overflow-y-auto" style={{ maxHeight: PANEL_MAX_HEIGHT - COLLAPSED_HEIGHT }}>
          <ControlUnitControls index={unitIndex} compact />
        </div>
      )}
    </div>
  );
}

/** Container that renders panels for all control units */
interface ControlFramePanelsProps {
  layout: CanvasLayout;
}

export function ControlFramePanels({ layout }: ControlFramePanelsProps) {
  const panelCollapsedOverrides = useCanvasStore((s) => s.panelCollapsedOverrides);
  const units = useControlStore((s) => s.units);

  // Only show panels if there are control frames in the layout
  if (layout.controlFrames.length === 0) return null;

  return (
    <>
      {units.map((unit, index) => {
        const hasImage = unit.image !== null;
        const override = panelCollapsedOverrides.get(index);
        // User override takes precedence, otherwise: collapsed if no image, expanded if has image
        const isCollapsed = override !== undefined ? override : !hasImage;

        return (
          <ControlFramePanel
            key={index}
            unitIndex={index}
            collapsed={isCollapsed}
            layout={layout}
          />
        );
      })}
    </>
  );
}
