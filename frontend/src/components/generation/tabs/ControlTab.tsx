import { useCallback } from "react";
import { useControlStore } from "@/stores/controlStore";
import { useCanvasStore } from "@/stores/canvasStore";
import { ParamSlider } from "../ParamSlider";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Combobox } from "@/components/ui/combobox";
import { Plus, Trash2, PenLine } from "lucide-react";
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

export function ControlTab() {
  const units = useControlStore((s) => s.units);
  const addUnit = useControlStore((s) => s.addUnit);
  const setUnitCount = useControlStore((s) => s.setUnitCount);

  return (
    <div className="flex flex-col gap-3 text-sm">
      <ParamSlider label="Units" value={units.length} onChange={setUnitCount} min={1} max={10} />

      <div className="flex flex-col gap-1.5">
        {units.map((_, i) => (
          <ControlUnitRow key={i} index={i} canRemove={units.length > 1} />
        ))}
      </div>

      <Button variant="outline" size="sm" className="w-full" onClick={addUnit} disabled={units.length >= 10}>
        <Plus size={12} className="mr-1" /> Add Unit
      </Button>
    </div>
  );
}

interface ControlUnitRowProps {
  index: number;
  canRemove: boolean;
}

function ControlUnitRow({ index, canRemove }: ControlUnitRowProps) {
  const unit = useControlStore((s) => s.units[index]);
  const setUnitParam = useControlStore((s) => s.setUnitParam);
  const setUnitType = useControlStore((s) => s.setUnitType);
  const removeUnit = useControlStore((s) => s.removeUnit);
  const toggleSeparateImage = useControlStore((s) => s.toggleSeparateImage);
  const setSelectedControlFrame = useCanvasStore((s) => s.setSelectedControlFrame);

  const handleEditOnCanvas = useCallback(() => {
    setSelectedControlFrame(index);
  }, [setSelectedControlFrame, index]);

  return (
    <div className="flex flex-col gap-1.5 p-2 rounded-md border border-border">
      {/* Row 1: Index + Type + Enabled + Remove */}
      <div className="flex items-center gap-1.5">
        <span className="text-[11px] text-muted-foreground font-mono w-4 flex-shrink-0">{index}</span>
        <Combobox
          value={unit.unitType}
          onValueChange={(v) => setUnitType(index, v as ControlUnitType)}
          options={UNIT_TYPE_OPTIONS}
          className="h-7 text-xs flex-1"
        />
        <div className="flex items-center gap-1">
          <Label className="text-[10px] text-muted-foreground">On</Label>
          <Switch checked={unit.enabled} onCheckedChange={(checked) => setUnitParam(index, "enabled", checked)} />
        </div>
        {canRemove && (
          <Button variant="ghost" size="icon-sm" onClick={() => removeUnit(index)} title="Remove unit">
            <Trash2 size={12} />
          </Button>
        )}
      </div>

      {/* Row 2: Separate image toggle + Edit on Canvas */}
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-1.5">
          <Switch
            checked={unit.useSeparateImage}
            onCheckedChange={() => toggleSeparateImage(index)}
          />
          <Label className="text-[10px] text-muted-foreground">Separate image</Label>
        </div>
        {unit.enabled && unit.useSeparateImage && (
          <Button
            variant="link"
            size="sm"
            className="h-auto p-0 text-[11px] text-amber-500 gap-1"
            onClick={handleEditOnCanvas}
          >
            <PenLine size={10} />
            Edit on Canvas
          </Button>
        )}
      </div>
    </div>
  );
}
