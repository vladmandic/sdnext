import { useCallback, useMemo } from "react";
import { useControlStore, resolveUnitImage } from "@/stores/controlStore";
import { useControlModels, useControlModes, usePreprocessImage, usePreprocessors } from "@/api/hooks/useControl";
import { useIPAdapterModels } from "@/api/hooks/useAdapters";
import { uploadFile } from "@/lib/upload";
import { ParamSlider } from "../../ParamSlider";
import { ParamSection } from "../../ParamSection";
import { ParamGrid } from "../../ParamRow";
import { ImageUpload } from "../../ImageUpload";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { X, Play, Loader2 } from "lucide-react";
import { Combobox } from "@/components/ui/combobox";
import { toast } from "sonner";

/** Infer reasonable slider range from a default value when the API provides no metadata. */
function inferSliderRange(defaultValue: number): { min: number; max: number; step: number } {
  if (Number.isInteger(defaultValue)) {
    if (defaultValue <= 1) return { min: 0, max: 10, step: 1 };
    if (defaultValue <= 64) return { min: 0, max: Math.max(512, defaultValue * 4), step: 1 };
    if (defaultValue <= 512) return { min: 0, max: Math.max(2048, defaultValue * 4), step: 1 };
    return { min: 0, max: defaultValue * 4, step: 1 };
  }
  // Float
  if (defaultValue <= 1) return { min: 0, max: 2, step: 0.01 };
  return { min: 0, max: Math.max(10, defaultValue * 4), step: 0.01 };
}

interface ControlUnitControlsProps {
  index: number;
  compact?: boolean;
}

export function ControlUnitControls({ index, compact }: ControlUnitControlsProps) {
  const unit = useControlStore((s) => s.units[index]);
  const units = useControlStore((s) => s.units);
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

  const resolvedImage = resolveUnitImage(units, index);

  const handleProcess = useCallback(async () => {
    if (!resolvedImage || unit.processor === "None") return;
    try {
      const ref = await uploadFile(resolvedImage);
      const params = Object.keys(unit.processorParams).length > 0 ? unit.processorParams : undefined;
      const result = await preprocessMutation.mutateAsync({ image: ref, model: unit.processor, params });
      setUnitParam(index, "processedImage", `data:image/png;base64,${result.image}`);
    } catch (err) {
      toast.error("Preprocessing failed", { description: err instanceof Error ? err.message : String(err) });
    }
  }, [resolvedImage, unit.processor, unit.processorParams, preprocessMutation, setUnitParam, index]);

  // Default params for the currently selected processor
  const processorDefaults = useMemo(() => {
    if (!preprocessors || unit.processor === "None") return null;
    const info = preprocessors.find((p) => p.name === unit.processor);
    if (!info || Object.keys(info.params).length === 0) return null;
    return info.params;
  }, [preprocessors, unit.processor]);

  const handleProcessorChange = useCallback((v: string) => {
    setUnitParam(index, "processor", v);
    setUnitParam(index, "processedImage", null);
    // Initialize params from API defaults for the new processor
    const info = preprocessors?.find((p) => p.name === v);
    setUnitParam(index, "processorParams", info?.params && Object.keys(info.params).length > 0 ? { ...info.params } : {});
  }, [index, setUnitParam, preprocessors]);

  const handleParamChange = useCallback((key: string, value: unknown) => {
    setUnitParam(index, "processorParams", { ...unit.processorParams, [key]: value });
  }, [index, setUnitParam, unit.processorParams]);

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
          <Label className="text-2xs text-muted-foreground w-16 flex-shrink-0">Processor</Label>
          <Combobox
            value={unit.processor}
            onValueChange={handleProcessorChange}
            options={["None", ...(preprocessors?.filter((p) => p.name !== "None").map((p) => p.name) ?? [])]}
            className="h-6 text-2xs flex-1"
          />
        </div>
      )}

      {/* Model */}
      {showModel && (
        <div className="flex items-center gap-2">
          <Label className="text-2xs text-muted-foreground w-16 flex-shrink-0">Model</Label>
          <Combobox
            value={unit.model}
            onValueChange={(v) => setUnitParam(index, "model", v)}
            options={["None", ...(models ?? [])]}
            className="h-6 text-2xs flex-1"
          />
        </div>
      )}

      {/* Mode */}
      {showModel && modesForModel && modesForModel.length > 0 && (
        <div className="flex items-center gap-2">
          <Label className="text-2xs text-muted-foreground w-16 flex-shrink-0">Mode</Label>
          <Combobox
            value={unit.mode}
            onValueChange={(v) => setUnitParam(index, "mode", v)}
            options={modesForModel}
            className="h-6 text-2xs flex-1"
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
            <Label className="text-2xs text-muted-foreground w-16 flex-shrink-0">Adapter</Label>
            <Combobox
              value={unit.adapter}
              onValueChange={(v) => setUnitParam(index, "adapter", v)}
              options={["None", ...(adapterModels ?? [])]}
              className="h-6 text-2xs flex-1"
            />
          </div>
          <ParamSlider label="Scale" value={unit.scale} onChange={(v) => setUnitParam(index, "scale", v)} min={0} max={2} step={0.01} />
          <div className="flex items-center gap-2">
            <Label className="text-2xs text-muted-foreground w-16 flex-shrink-0">Crop</Label>
            <Switch checked={unit.crop} onCheckedChange={(checked) => setUnitParam(index, "crop", checked)} />
          </div>
        </>
      )}

      {/* Timing */}
      {showTiming && (
        <ParamSection title="Timing" defaultOpen={false}>
          <ParamGrid>
            <ParamSlider label="Start" value={unit.start} onChange={(v) => setUnitParam(index, "start", v)} min={0} max={1} step={0.01} />
            <ParamSlider label="End" value={unit.end} onChange={(v) => setUnitParam(index, "end", v)} min={0} max={1} step={0.01} />
          </ParamGrid>
        </ParamSection>
      )}

      {/* Processor params */}
      {showProcessor && processorDefaults && (
        <ParamSection title="Processor Settings" defaultOpen={false}>
          {Object.entries(unit.processorParams).map(([key, value]) => {
            const def = processorDefaults[key];
            if (typeof value === "boolean" || typeof def === "boolean") {
              return (
                <label key={key} className="flex items-center gap-1.5 text-2xs text-muted-foreground cursor-pointer">
                  <Checkbox checked={!!value} onCheckedChange={(c) => handleParamChange(key, !!c)} />
                  {key}
                </label>
              );
            }
            if (typeof value === "number" || typeof def === "number") {
              const numDef = typeof def === "number" ? def : (typeof value === "number" ? value : 0);
              const inferred = inferSliderRange(numDef);
              return <ParamSlider key={key} label={key} value={typeof value === "number" ? value : numDef} onChange={(v) => handleParamChange(key, v)} min={inferred.min} max={inferred.max} step={inferred.step} />;
            }
            return null;
          })}
        </ParamSection>
      )}

      {/* Guess mode + process button — merged row in compact mode */}
      {compact ? (
        <div className="flex items-center justify-between">
          {showGuess && (
            <label className="flex items-center gap-1.5 text-2xs text-muted-foreground cursor-pointer">
              <Checkbox checked={unit.guess} onCheckedChange={(c) => setUnitParam(index, "guess", !!c)} />
              Guess mode
            </label>
          )}
          {showControlImage && showProcessor && resolvedImage && unit.processor !== "None" && (
            <Button
              variant="outline"
              size="sm"
              className="h-6 text-2xs px-2 gap-1 ml-auto"
              onClick={handleProcess}
              disabled={preprocessMutation.isPending}
            >
              {preprocessMutation.isPending ? <Loader2 size={10} className="animate-spin" /> : <Play size={10} />}
              Process
            </Button>
          )}
        </div>
      ) : (
        <>
          {/* Guess mode */}
          {showGuess && (
            <label className="flex items-center gap-1.5 text-2xs text-muted-foreground cursor-pointer px-1">
              <Checkbox checked={unit.guess} onCheckedChange={(c) => setUnitParam(index, "guess", !!c)} />
              Guess mode
            </label>
          )}

          {/* Control image + process button */}
          {showControlImage && (
            <div className="flex flex-col gap-1">
              <div className="flex items-center justify-between">
                <Label className="text-2xs text-muted-foreground">Control Image</Label>
                {showProcessor && resolvedImage && unit.processor !== "None" && (
                  <Button
                    variant="outline"
                    size="sm"
                    className="h-6 text-2xs px-2 gap-1"
                    onClick={handleProcess}
                    disabled={preprocessMutation.isPending}
                  >
                    {preprocessMutation.isPending ? <Loader2 size={10} className="animate-spin" /> : <Play size={10} />}
                    Process
                  </Button>
                )}
              </div>
              {unit.processedImage && (
                <button
                  type="button"
                  onClick={() => setUnitParam(index, "processedImage", null)}
                  className="text-3xs text-muted-foreground hover:text-destructive transition-colors flex items-center gap-1"
                >
                  <X size={10} />
                  Clear processed
                </button>
              )}
            </div>
          )}
        </>
      )}

      {/* Factor */}
      {showFactor && (
        <ParamSlider label="Factor" value={unit.factor} onChange={(v) => setUnitParam(index, "factor", v)} min={0.01} max={2} step={0.01} />
      )}

      {/* Reference fields */}
      {showReference && (
        <>
          <div className="flex items-center gap-2">
            <Label className="text-2xs text-muted-foreground w-16 flex-shrink-0">Attention</Label>
            <Combobox
              value={unit.attention}
              onValueChange={(v) => setUnitParam(index, "attention", v)}
              options={["Attention", "Adain", "Attention and Adain"]}
              className="h-6 text-2xs flex-1"
            />
          </div>
          <ParamGrid>
            <ParamSlider label="Query Weight" value={unit.queryWeight} onChange={(v) => setUnitParam(index, "queryWeight", v)} min={0} max={2} step={0.01} />
            <ParamSlider label="Adain Weight" value={unit.adainWeight} onChange={(v) => setUnitParam(index, "adainWeight", v)} min={0} max={2} step={0.01} />
          </ParamGrid>
          <ParamSlider label="Fidelity" value={unit.fidelity} onChange={(v) => setUnitParam(index, "fidelity", v)} min={0} max={1} step={0.01} />
        </>
      )}

      {/* IP-Adapter images */}
      {showIPAdapter && (
        <div className="flex flex-col gap-1">
          <Label className="text-2xs text-muted-foreground">Images</Label>
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
          <Label className="text-2xs text-muted-foreground">Masks</Label>
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
