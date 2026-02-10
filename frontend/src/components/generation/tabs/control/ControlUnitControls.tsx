import { useCallback, useMemo } from "react";
import { useControlStore } from "@/stores/controlStore";
import { useControlModels, useControlModes, usePreprocessImage, usePreprocessors } from "@/api/hooks/useControl";
import { useIPAdapterModels } from "@/api/hooks/useAdapters";
import { fileToBase64 } from "@/lib/image";
import { ParamSlider } from "../../ParamSlider";
import { ParamSection } from "../../ParamSection";
import { ImageUpload } from "../../ImageUpload";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { X, Play, Loader2 } from "lucide-react";
import { Combobox } from "@/components/ui/combobox";
import { toast } from "sonner";

interface ControlUnitControlsProps {
  index: number;
  compact?: boolean;
}

export function ControlUnitControls({ index, compact }: ControlUnitControlsProps) {
  const unit = useControlStore((s) => s.units[index]);
  const setUnitParam = useControlStore((s) => s.setUnitParam);
  const addUnitImage = useControlStore((s) => s.addUnitImage);
  const removeUnitImage = useControlStore((s) => s.removeUnitImage);
  const addUnitMask = useControlStore((s) => s.addUnitMask);
  const removeUnitMask = useControlStore((s) => s.removeUnitMask);
  const { data: models } = useControlModels(unit.unitType);
  const { data: controlModes } = useControlModes();
  const { data: preprocessors } = usePreprocessors();
  const { data: adapterModels } = useIPAdapterModels();
  const preprocessMutation = usePreprocessImage();

  const handleProcess = useCallback(async () => {
    if (!unit.image || unit.processor === "None") return;
    try {
      const b64 = await fileToBase64(unit.image);
      const result = await preprocessMutation.mutateAsync({ image: b64, model: unit.processor });
      setUnitParam(index, "processedImage", `data:image/png;base64,${result.image}`);
    } catch (err) {
      toast.error("Preprocessing failed", { description: err instanceof Error ? err.message : String(err) });
    }
  }, [unit.image, unit.processor, preprocessMutation, setUnitParam, index]);

  const modesForModel = useMemo(() => {
    if (!controlModes || unit.model === "None") return null;
    if (controlModes[unit.model]) return controlModes[unit.model];
    for (const [key, modes] of Object.entries(controlModes)) {
      if (unit.model.toLowerCase().includes(key.toLowerCase())) return modes;
    }
    return null;
  }, [controlModes, unit.model]);

  const type = unit.unitType;
  const showProcessor = type !== "reference" && type !== "ip" && type !== "asset";
  const showModel = type !== "reference" && type !== "ip" && type !== "asset";
  const showTiming = type === "controlnet" || type === "xs" || type === "ip";
  const showGuess = type === "controlnet";
  const showFactor = type === "t2i";
  const showReference = type === "reference";
  const showIPAdapter = type === "ip";
  const showControlImage = type !== "ip";

  const imagePreviews = useMemo(() => unit.images.map((f) => URL.createObjectURL(f)), [unit.images]);
  const maskPreviews = useMemo(() => unit.masks.map((f) => URL.createObjectURL(f)), [unit.masks]);

  const gap = compact ? "gap-1.5" : "gap-2";

  return (
    <div className={`flex flex-col ${gap}`}>
      {/* Processor */}
      {showProcessor && (
        <div className="flex items-center gap-2">
          <Label className="text-[11px] text-muted-foreground w-16 flex-shrink-0">Processor</Label>
          <Combobox
            value={unit.processor}
            onValueChange={(v) => { setUnitParam(index, "processor", v); setUnitParam(index, "processedImage", null); }}
            options={["None", ...(preprocessors?.filter((p) => p.name !== "None").map((p) => p.name) ?? [])]}
            className="h-7 text-xs flex-1"
          />
        </div>
      )}

      {/* Model */}
      {showModel && (
        <div className="flex items-center gap-2">
          <Label className="text-[11px] text-muted-foreground w-16 flex-shrink-0">Model</Label>
          <Combobox
            value={unit.model}
            onValueChange={(v) => setUnitParam(index, "model", v)}
            options={["None", ...(models ?? [])]}
            className="h-7 text-xs flex-1"
          />
        </div>
      )}

      {/* Mode */}
      {showModel && modesForModel && modesForModel.length > 0 && (
        <div className="flex items-center gap-2">
          <Label className="text-[11px] text-muted-foreground w-16 flex-shrink-0">Mode</Label>
          <Combobox
            value={unit.mode}
            onValueChange={(v) => setUnitParam(index, "mode", v)}
            options={modesForModel}
            className="h-7 text-xs flex-1"
          />
        </div>
      )}

      {/* Strength */}
      {showModel && (
        <ParamSlider label="Strength" value={unit.strength} onChange={(v) => setUnitParam(index, "strength", v)} min={0.01} max={2} step={0.01} />
      )}

      {/* IP-Adapter fields */}
      {showIPAdapter && (
        <>
          <div className="flex items-center gap-2">
            <Label className="text-[11px] text-muted-foreground w-16 flex-shrink-0">Adapter</Label>
            <Combobox
              value={unit.adapter}
              onValueChange={(v) => setUnitParam(index, "adapter", v)}
              options={["None", ...(adapterModels ?? [])]}
              className="h-7 text-xs flex-1"
            />
          </div>
          <ParamSlider label="Scale" value={unit.scale} onChange={(v) => setUnitParam(index, "scale", v)} min={0} max={2} step={0.01} />
          <div className="flex items-center gap-2">
            <Label className="text-[11px] text-muted-foreground w-16 flex-shrink-0">Crop</Label>
            <Switch checked={unit.crop} onCheckedChange={(checked) => setUnitParam(index, "crop", checked)} />
          </div>
        </>
      )}

      {/* Timing */}
      {showTiming && (
        <ParamSection title="Timing" defaultOpen={false}>
          <ParamSlider label="Start" value={unit.start} onChange={(v) => setUnitParam(index, "start", v)} min={0} max={1} step={0.01} />
          <ParamSlider label="End" value={unit.end} onChange={(v) => setUnitParam(index, "end", v)} min={0} max={1} step={0.01} />
        </ParamSection>
      )}

      {/* Guess mode */}
      {showGuess && (
        <label className="flex items-center gap-1.5 text-[11px] text-muted-foreground cursor-pointer px-1">
          <Checkbox checked={unit.guess} onCheckedChange={(c) => setUnitParam(index, "guess", !!c)} />
          Guess mode
        </label>
      )}

      {/* Factor */}
      {showFactor && (
        <ParamSlider label="Factor" value={unit.factor} onChange={(v) => setUnitParam(index, "factor", v)} min={0.01} max={2} step={0.01} />
      )}

      {/* Reference fields */}
      {showReference && (
        <>
          <div className="flex items-center gap-2">
            <Label className="text-[11px] text-muted-foreground w-16 flex-shrink-0">Attention</Label>
            <Combobox
              value={unit.attention}
              onValueChange={(v) => setUnitParam(index, "attention", v)}
              options={["Attention", "Adain", "Attention and Adain"]}
              className="h-7 text-xs flex-1"
            />
          </div>
          <ParamSlider label="Fidelity" value={unit.fidelity} onChange={(v) => setUnitParam(index, "fidelity", v)} min={0} max={1} step={0.01} />
          <ParamSlider label="Query Weight" value={unit.queryWeight} onChange={(v) => setUnitParam(index, "queryWeight", v)} min={0} max={2} step={0.01} />
          <ParamSlider label="Adain Weight" value={unit.adainWeight} onChange={(v) => setUnitParam(index, "adainWeight", v)} min={0} max={2} step={0.01} />
        </>
      )}

      {/* Control image + process button */}
      {showControlImage && (
        <div className="flex flex-col gap-1">
          <div className="flex items-center justify-between">
            <Label className="text-[11px] text-muted-foreground">Control Image</Label>
            {showProcessor && unit.image && unit.processor !== "None" && (
              <Button
                variant="outline"
                size="sm"
                className="h-6 text-[11px] px-2 gap-1"
                onClick={handleProcess}
                disabled={preprocessMutation.isPending}
              >
                {preprocessMutation.isPending ? <Loader2 size={10} className="animate-spin" /> : <Play size={10} />}
                Process
              </Button>
            )}
          </div>
          {unit.processedImage && (
            <div>
              <div className="relative h-20 rounded-md overflow-hidden border border-border group">
                <img src={unit.processedImage} alt="Processed" className="w-full h-full object-cover" />
                <Button
                  variant="destructive"
                  size="icon-sm"
                  className="absolute top-1 right-1 opacity-0 group-hover:opacity-100 transition-opacity h-5 w-5"
                  onClick={() => setUnitParam(index, "processedImage", null)}
                >
                  <X size={10} />
                </Button>
              </div>
            </div>
          )}
        </div>
      )}

      {/* IP-Adapter images */}
      {showIPAdapter && (
        <div className="flex flex-col gap-1">
          <Label className="text-[11px] text-muted-foreground">Images</Label>
          <div className="flex gap-1 flex-wrap">
            {imagePreviews.map((url, i) => (
              <div key={i} className="relative h-16 w-16 rounded border border-border overflow-hidden group">
                <img src={url} alt={`ref ${i}`} className="w-full h-full object-cover" />
                <Button variant="destructive" size="icon-sm" className="absolute top-0 right-0 opacity-0 group-hover:opacity-100 h-4 w-4" onClick={() => removeUnitImage(index, i)}>
                  <X size={8} />
                </Button>
              </div>
            ))}
          </div>
          <ImageUpload image={null} onImageChange={(file) => { if (file) addUnitImage(index, file); }} label="Add reference" compact />
        </div>
      )}

      {/* IP-Adapter masks */}
      {showIPAdapter && (
        <div className="flex flex-col gap-1">
          <Label className="text-[11px] text-muted-foreground">Masks</Label>
          <div className="flex gap-1 flex-wrap">
            {maskPreviews.map((url, i) => (
              <div key={i} className="relative h-16 w-16 rounded border border-border overflow-hidden group">
                <img src={url} alt={`mask ${i}`} className="w-full h-full object-cover" />
                <Button variant="destructive" size="icon-sm" className="absolute top-0 right-0 opacity-0 group-hover:opacity-100 h-4 w-4" onClick={() => removeUnitMask(index, i)}>
                  <X size={8} />
                </Button>
              </div>
            ))}
          </div>
          <ImageUpload image={null} onImageChange={(file) => { if (file) addUnitMask(index, file); }} label="Add mask" compact />
        </div>
      )}
    </div>
  );
}
