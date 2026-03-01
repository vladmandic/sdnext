import { useCallback } from "react";
import { useCanvasStore } from "@/stores/canvasStore";
import { useImg2ImgStore } from "@/stores/img2imgStore";
import { useShortcut } from "@/hooks/useShortcut";
import { Move, Paintbrush, Eraser, Eye, EyeOff, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";

export function CanvasToolbar() {
  const activeTool = useCanvasStore((s) => s.activeTool);
  const setActiveTool = useCanvasStore((s) => s.setActiveTool);
  const brushSize = useCanvasStore((s) => s.brushSize);
  const setBrushSize = useCanvasStore((s) => s.setBrushSize);
  const maskVisible = useCanvasStore((s) => s.maskVisible);
  const setMaskVisible = useCanvasStore((s) => s.setMaskVisible);
  const clearMask = useImg2ImgStore((s) => s.clearMask);
  const maskLineCount = useImg2ImgStore((s) => s.maskLines.length);

  const toggleBrush = useCallback(() => {
    setActiveTool(activeTool === "maskBrush" ? "move" : "maskBrush");
  }, [activeTool, setActiveTool]);

  const toggleEraser = useCallback(() => {
    setActiveTool(activeTool === "maskEraser" ? "move" : "maskEraser");
  }, [activeTool, setActiveTool]);

  const handleSizeChange = useCallback(([v]: number[]) => setBrushSize(v), [setBrushSize]);

  // Keyboard shortcuts (dispatched via the global shortcut system, scoped to "canvas")
  useShortcut("canvas-move", () => setActiveTool("move"));
  useShortcut("canvas-brush", () => {
    const tool = useCanvasStore.getState().activeTool;
    setActiveTool(tool === "maskBrush" ? "move" : "maskBrush");
  });
  useShortcut("canvas-eraser", () => {
    const tool = useCanvasStore.getState().activeTool;
    setActiveTool(tool === "maskEraser" ? "move" : "maskEraser");
  });
  useShortcut("canvas-deselect", () => setActiveTool("move"));
  useShortcut("canvas-brush-smaller", () => {
    setBrushSize(Math.max(1, useCanvasStore.getState().brushSize - 5));
  });
  useShortcut("canvas-brush-larger", () => {
    setBrushSize(Math.min(200, useCanvasStore.getState().brushSize + 5));
  });

  return (
    <div className="absolute bottom-3 left-1/2 -translate-x-1/2 flex items-center gap-2 px-3 py-1.5 rounded-lg bg-background/80 backdrop-blur-sm border border-border shadow-lg">
      {/* Move */}
      <Button
        variant={activeTool === "move" ? "default" : "ghost"}
        size="icon-xs"
        onClick={() => setActiveTool("move")}
        title="Move / select layers (V)"
      >
        <Move size={14} />
      </Button>

      {/* Brush */}
      <Button
        variant={activeTool === "maskBrush" ? "default" : "ghost"}
        size="icon-xs"
        onClick={toggleBrush}
        title="Paint inpainting mask (B)"
      >
        <Paintbrush size={14} />
      </Button>

      {/* Eraser */}
      <Button
        variant={activeTool === "maskEraser" ? "default" : "ghost"}
        size="icon-xs"
        onClick={toggleEraser}
        title="Erase inpainting mask (E)"
      >
        <Eraser size={14} />
      </Button>

      {/* Separator */}
      <div className="w-px h-5 bg-border" />

      {/* Brush size */}
      <Label className="text-3xs text-muted-foreground whitespace-nowrap">{brushSize}px</Label>
      <Slider
        min={1}
        max={200}
        step={1}
        value={[brushSize]}
        onValueChange={handleSizeChange}
        className="w-24"
      />

      {/* Separator */}
      <div className="w-px h-5 bg-border" />

      {/* Mask visibility */}
      <Button
        variant="ghost"
        size="icon-xs"
        onClick={() => setMaskVisible(!maskVisible)}
        title={maskVisible ? "Hide mask" : "Show mask"}
      >
        {maskVisible ? <Eye size={14} /> : <EyeOff size={14} />}
      </Button>

      {/* Clear mask */}
      <Button
        variant="ghost"
        size="icon-xs"
        onClick={clearMask}
        disabled={maskLineCount === 0}
        title="Clear all mask strokes"
      >
        <Trash2 size={14} />
      </Button>
    </div>
  );
}
