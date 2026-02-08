import { useMemo } from "react";
import { useControlStore } from "@/stores/controlStore";
import { useControlModels, usePreprocessors } from "@/api/hooks/useControl";
import { useIPAdapterModels } from "@/api/hooks/useAdapters";
import { ParamSlider } from "../../ParamSlider";
import { ParamSection } from "../../ParamSection";
import { ImageUpload } from "../../ImageUpload";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Trash2, X } from "lucide-react";
import { Combobox } from "@/components/ui/combobox";
import type { ControlUnitType } from "@/api/types/control";

const UNIT_TYPE_OPTIONS: { value: ControlUnitType; label: string }[] = [
  { value: "asset", label: "Asset" },
  { value: "controlnet", label: "ControlNet" },
  { value: "t2i", label: "T2I-Adapter" },
  { value: "xs", label: "XS" },
  { value: "lite", label: "Lite" },
  { value: "reference", label: "Reference" },
  { value: "ip", label: "IP-Adapter" },
];

interface ControlUnitCardProps {
  index: number;
  canRemove: boolean;
}

export function ControlUnitCard({ index, canRemove }: ControlUnitCardProps) {
  const unit = useControlStore((s) => s.units[index]);
  const setUnitParam = useControlStore((s) => s.setUnitParam);
  const setUnitImage = useControlStore((s) => s.setUnitImage);
  const setUnitType = useControlStore((s) => s.setUnitType);
  const removeUnit = useControlStore((s) => s.removeUnit);
  const addUnitImage = useControlStore((s) => s.addUnitImage);
  const removeUnitImage = useControlStore((s) => s.removeUnitImage);
  const addUnitMask = useControlStore((s) => s.addUnitMask);
  const removeUnitMask = useControlStore((s) => s.removeUnitMask);
  const { data: models } = useControlModels(unit.unitType);
  const { data: preprocessors } = usePreprocessors();
  const { data: adapterModels } = useIPAdapterModels();

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

  return (
    <div className="flex flex-col gap-2 p-2 rounded-md border border-border">
      {/* Header: Type + Enabled + Remove */}
      <div className="flex items-center justify-between gap-2">
        <Combobox
          value={unit.unitType}
          onValueChange={(v) => setUnitType(index, v as ControlUnitType)}
          options={UNIT_TYPE_OPTIONS}
          className="h-7 text-xs w-28 flex-shrink-0"
        />
        <div className="flex items-center gap-2">
          <Label className="text-[11px] text-muted-foreground">Enabled</Label>
          <Switch checked={unit.enabled} onCheckedChange={(checked) => setUnitParam(index, "enabled", checked)} />
        </div>
        {canRemove && (
          <Button variant="ghost" size="icon-sm" onClick={() => removeUnit(index)} title="Remove unit">
            <Trash2 size={12} />
          </Button>
        )}
      </div>

      {/* Processor (not for Reference or IP-Adapter) */}
      {showProcessor && (
        <div className="flex items-center gap-2">
          <Label className="text-[11px] text-muted-foreground w-16 flex-shrink-0">Processor</Label>
          <Combobox
            value={unit.processor}
            onValueChange={(v) => setUnitParam(index, "processor", v)}
            options={["None", ...(preprocessors?.filter((p) => p.name !== "None").map((p) => p.name) ?? [])]}
            className="h-7 text-xs flex-1"
          />
        </div>
      )}

      {/* Model (not for Reference or IP-Adapter) */}
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

      {/* Strength (not for Reference or IP-Adapter) */}
      {showModel && (
        <ParamSlider label="Strength" value={unit.strength} onChange={(v) => setUnitParam(index, "strength", v)} min={0.01} max={2} step={0.01} />
      )}

      {/* IP-Adapter-specific fields */}
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

      {/* Timing: ControlNet, XS, and IP-Adapter */}
      {showTiming && (
        <ParamSection title="Timing" defaultOpen={false}>
          <ParamSlider label="Start" value={unit.start} onChange={(v) => setUnitParam(index, "start", v)} min={0} max={1} step={0.01} />
          <ParamSlider label="End" value={unit.end} onChange={(v) => setUnitParam(index, "end", v)} min={0} max={1} step={0.01} />
        </ParamSection>
      )}

      {/* Guess mode: ControlNet only */}
      {showGuess && (
        <label className="flex items-center gap-1.5 text-[11px] text-muted-foreground cursor-pointer px-1">
          <Checkbox checked={unit.guess} onCheckedChange={(c) => setUnitParam(index, "guess", !!c)} />
          Guess mode
        </label>
      )}

      {/* Factor: T2I-Adapter only */}
      {showFactor && (
        <ParamSlider label="Factor" value={unit.factor} onChange={(v) => setUnitParam(index, "factor", v)} min={0.01} max={2} step={0.01} />
      )}

      {/* Reference-specific fields */}
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

      {/* Control Image: all types except IP-Adapter */}
      {showControlImage && (
        <div className="flex flex-col gap-1">
          <Label className="text-[11px] text-muted-foreground">Control Image</Label>
          <ImageUpload
            image={unit.image}
            onImageChange={(file) => setUnitImage(index, file)}
            label="Drop control image"
            compact
          />
        </div>
      )}

      {/* IP-Adapter: Reference Images */}
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

      {/* IP-Adapter: Masks */}
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
