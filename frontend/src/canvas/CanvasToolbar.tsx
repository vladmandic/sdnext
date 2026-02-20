import { useEffect, useCallback } from "react";
import { useCanvasStore } from "@/stores/canvasStore";
import { useImg2ImgStore } from "@/stores/img2imgStore";
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

  // Keyboard shortcuts
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      // Skip if typing in an input
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;

      switch (e.key.toLowerCase()) {
        case "v":
          setActiveTool("move");
          break;
        case "b":
          setActiveTool(activeTool === "maskBrush" ? "move" : "maskBrush");
          break;
        case "e":
          setActiveTool(activeTool === "maskEraser" ? "move" : "maskEraser");
          break;
        case "escape":
          setActiveTool("move");
          break;
        case "[":
          setBrushSize(Math.max(1, brushSize - 5));
          break;
        case "]":
          setBrushSize(Math.min(200, brushSize + 5));
          break;
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [activeTool, brushSize, setActiveTool, setBrushSize]);

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
      <Label className="text-[10px] text-muted-foreground whitespace-nowrap">{brushSize}px</Label>
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
