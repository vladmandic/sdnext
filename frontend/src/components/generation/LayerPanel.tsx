import { useCallback, useRef } from "react";
import { useCanvasStore, type ImageLayer, type MaskObjectLayer } from "@/stores/canvasStore";
import { useGenerationStore } from "@/stores/generationStore";
import { fileToBase64 } from "@/lib/image";
import { Eye, EyeOff, X, Plus, Frame, Lock, Unlock } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

function LayerDims({ layer }: { layer: ImageLayer }) {
  const canvasW = Math.round(Math.abs(layer.naturalWidth * layer.scaleX));
  const canvasH = Math.round(Math.abs(layer.naturalHeight * layer.scaleY));
  const isTransformed = canvasW !== layer.naturalWidth || canvasH !== layer.naturalHeight;
  return (
    <p className="text-4xs text-muted-foreground" title={`Native: ${layer.naturalWidth}×${layer.naturalHeight}`}>
      {isTransformed ? <>{canvasW}&times;{canvasH} <span className="opacity-50">({layer.naturalWidth}&times;{layer.naturalHeight})</span></> : <>{layer.naturalWidth}&times;{layer.naturalHeight}</>}
    </p>
  );
}

export function LayerPanel() {
  const layers = useCanvasStore((s) => s.layers);
  const activeLayerId = useCanvasStore((s) => s.activeLayerId);
  const setActiveLayer = useCanvasStore((s) => s.setActiveLayer);
  const updateLayer = useCanvasStore((s) => s.updateLayer);
  const removeLayer = useCanvasStore((s) => s.removeLayer);
  const addImageLayer = useCanvasStore((s) => s.addImageLayer);
  const setParam = useGenerationStore((s) => s.setParam);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const imageLayers = layers.filter((l) => l.type === "image") as ImageLayer[];
  const maskLayers = layers.filter((l) => l.type === "mask") as MaskObjectLayer[];

  const handleAddFile = useCallback(async (file: File) => {
    if (!file.type.startsWith("image/")) return;
    const base64 = await fileToBase64(file);
    const objectUrl = URL.createObjectURL(file);
    const img = new window.Image();
    img.src = objectUrl;
    await new Promise<void>((r) => { img.onload = () => r(); });
    addImageLayer(file, base64, objectUrl, img.naturalWidth, img.naturalHeight);
  }, [addImageLayer]);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files) {
      for (const file of files) handleAddFile(file);
    }
    e.target.value = "";
  }, [handleAddFile]);

  const openPicker = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  return (
    <div className="flex flex-col gap-1">
      {imageLayers.length === 0 && (
        <p className="text-3xs text-muted-foreground text-center py-1">Drop images onto canvas</p>
      )}

      {imageLayers.map((layer) => {
        const isActive = layer.id === activeLayerId;
        const truncName = layer.name.length > 18 ? `${layer.name.slice(0, 15)}...` : layer.name;
        return (
          <div
            key={layer.id}
            className={cn(
              "flex items-center gap-1.5 px-1.5 py-1 rounded cursor-pointer transition-colors",
              isActive ? "bg-primary/15 border border-primary/30" : "hover:bg-muted/60 border border-transparent",
            )}
            onClick={() => setActiveLayer(layer.id)}
          >
            {/* Thumbnail */}
            <img
              src={layer.imageData}
              alt={layer.name}
              className="w-8 h-8 rounded object-cover flex-shrink-0"
            />
            {/* Name + dims */}
            <div className="flex-1 min-w-0">
              <p className="text-3xs truncate" title={layer.name}>{truncName}</p>
              <LayerDims layer={layer} />
            </div>
            {/* Fit frame to image */}
            <Button
              variant="ghost"
              size="icon-xs"
              onClick={(e) => {
                e.stopPropagation();
                const w = Math.max(64, Math.round(Math.abs(layer.naturalWidth * layer.scaleX) / 8) * 8);
                const h = Math.max(64, Math.round(Math.abs(layer.naturalHeight * layer.scaleY) / 8) * 8);
                setParam("width", w);
                setParam("height", h);
                updateLayer(layer.id, {
                  x: (w - layer.naturalWidth * layer.scaleX) / 2,
                  y: (h - layer.naturalHeight * layer.scaleY) / 2,
                } as Partial<ImageLayer>);
              }}
              className="text-muted-foreground flex-shrink-0"
              title="Fit frame to this layer and center it (snapped to 8px grid)"
            >
              <Frame size={10} />
            </Button>
            {/* Visibility toggle */}
            <Button
              variant="ghost"
              size="icon-xs"
              onClick={(e) => {
                e.stopPropagation();
                updateLayer(layer.id, { visible: !layer.visible });
              }}
              className="text-muted-foreground flex-shrink-0"
              title={layer.visible ? "Hide layer — hidden layers are excluded from the composite sent to the backend" : "Show layer"}
            >
              {layer.visible ? <Eye size={10} /> : <EyeOff size={10} />}
            </Button>
            {/* Delete */}
            <Button
              variant="ghost"
              size="icon-xs"
              onClick={(e) => {
                e.stopPropagation();
                removeLayer(layer.id);
              }}
              className="text-muted-foreground flex-shrink-0"
              title="Remove layer"
            >
              <X size={10} />
            </Button>
          </div>
        );
      })}

      {/* Mask objects */}
      {maskLayers.length > 0 && (
        <>
          <p className="text-4xs text-muted-foreground uppercase tracking-wider px-1 pt-1">Masks</p>
          {maskLayers.map((layer) => {
            const isActive = layer.id === activeLayerId;
            return (
              <div
                key={layer.id}
                className={cn(
                  "flex items-center gap-1.5 px-1.5 py-1 rounded cursor-pointer transition-colors",
                  isActive ? "bg-primary/15 border border-primary/30" : "hover:bg-muted/60 border border-transparent",
                )}
                onClick={() => setActiveLayer(layer.id)}
              >
                <img
                  src={layer.imageData}
                  alt={layer.name}
                  className="w-8 h-8 rounded object-cover flex-shrink-0 bg-black/20"
                />
                <div className="flex-1 min-w-0">
                  <p className="text-3xs truncate" title={layer.name}>{layer.name}</p>
                  <p className="text-4xs text-muted-foreground">{layer.width}&times;{layer.height}</p>
                </div>
                {/* Lock toggle */}
                <Button
                  variant="ghost"
                  size="icon-xs"
                  onClick={(e) => {
                    e.stopPropagation();
                    updateLayer(layer.id, { locked: !layer.locked });
                  }}
                  className="text-muted-foreground flex-shrink-0"
                  title={layer.locked ? "Unlock mask — allows moving and resizing" : "Lock mask — prevents accidental interaction"}
                >
                  {layer.locked ? <Lock size={10} /> : <Unlock size={10} />}
                </Button>
                {/* Visibility toggle */}
                <Button
                  variant="ghost"
                  size="icon-xs"
                  onClick={(e) => {
                    e.stopPropagation();
                    updateLayer(layer.id, { visible: !layer.visible });
                  }}
                  className="text-muted-foreground flex-shrink-0"
                  title={layer.visible ? "Hide mask" : "Show mask"}
                >
                  {layer.visible ? <Eye size={10} /> : <EyeOff size={10} />}
                </Button>
                {/* Delete */}
                <Button
                  variant="ghost"
                  size="icon-xs"
                  onClick={(e) => {
                    e.stopPropagation();
                    removeLayer(layer.id);
                  }}
                  className="text-muted-foreground flex-shrink-0"
                  title="Remove mask"
                >
                  <X size={10} />
                </Button>
              </div>
            );
          })}
        </>
      )}

      {/* Add image button */}
      <Button
        variant="ghost"
        size="sm"
        onClick={openPicker}
        className="w-full h-6 text-3xs text-muted-foreground"
        title="Add an image layer to the canvas. You can also drag and drop files directly onto the canvas."
      >
        <Plus size={10} />
        Add Image
      </Button>
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        multiple
        onChange={handleFileInput}
        className="hidden"
      />
    </div>
  );
}
