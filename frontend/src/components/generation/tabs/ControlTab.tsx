import { useCallback, useMemo, useState } from "react";
import { useControlStore } from "@/stores/controlStore";
import { useCanvasStore } from "@/stores/canvasStore";
import { useUiStore } from "@/stores/uiStore";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Combobox } from "@/components/ui/combobox";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Plus, Trash2, PenLine } from "lucide-react";
import type { ControlUnitType } from "@/api/types/control";
import { UNIT_TYPE_LABELS, EXCLUSIVE_CONTROL_TYPES } from "@/api/types/control";
import { useUnifiedInputs } from "@/hooks/useUnifiedInputs";

const UNIT_TYPE_OPTIONS: { value: ControlUnitType; label: string }[] = (Object.entries(UNIT_TYPE_LABELS) as [ControlUnitType, string][]).map(([value, label]) => ({ value, label }));

function CanvasInputRow() {
  const inputRole = useCanvasStore((s) => s.inputRole);
  const label = inputRole === "reference" ? "Reference" : "Initial";
  return (
    <div className="flex items-center gap-1.5 p-2 rounded-md border border-border">
      <span className="text-2xs text-muted-foreground font-mono w-4 shrink-0">1</span>
      <span className="text-2xs flex-1">{label}</span>
      <span className="text-2xs text-muted-foreground">Input frame</span>
    </div>
  );
}

interface AddInputPopoverProps {
  availableSubTypes: { value: ControlUnitType; label: string; disabled: boolean }[];
  onAdd: (unitType: ControlUnitType) => void;
  disabled: boolean;
}

function AddInputPopover({ availableSubTypes, onAdd, disabled }: AddInputPopoverProps) {
  const [open, setOpen] = useState(false);

  const handleAdd = useCallback((unitType: ControlUnitType) => {
    onAdd(unitType);
    setOpen(false);
  }, [onAdd]);

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button variant="outline" size="sm" className="w-full" disabled={disabled}>
          <Plus size={12} className="mr-1" /> Add Input
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-56 p-1" align="start">
        <button
          disabled
          className="flex w-full items-center rounded-sm px-2 py-1.5 text-sm opacity-40 cursor-not-allowed"
          title="Input 1 is always available as Initial"
        >
          Initial
        </button>
        <button
          className="flex w-full items-center rounded-sm px-2 py-1.5 text-sm hover:bg-accent hover:text-accent-foreground"
          onClick={() => handleAdd("reference")}
        >
          Reference
        </button>
        <div className="px-2 py-1.5 text-xs font-medium text-muted-foreground">Control</div>
        {availableSubTypes.map((st) => (
          <button
            key={st.value}
            disabled={st.disabled}
            className={`flex w-full items-center rounded-sm px-2 py-1.5 text-sm pl-4 ${st.disabled ? "opacity-40 cursor-not-allowed" : "hover:bg-accent hover:text-accent-foreground"}`}
            onClick={() => !st.disabled && handleAdd(st.value)}
          >
            {st.label}
          </button>
        ))}
      </PopoverContent>
    </Popover>
  );
}

export function ControlTab() {
  const { availableControlSubTypes } = useUnifiedInputs();
  const units = useControlStore((s) => s.units);
  const addUnitWithType = useControlStore((s) => s.addUnitWithType);
  const reprocessOnGenerate = useUiStore((s) => s.reprocessOnGenerate);
  const setAutoUpdateProcessed = useUiStore((s) => s.setAutoUpdateProcessed);

  return (
    <div className="flex flex-col gap-3 text-sm">
      <div className="flex items-center justify-between">
        <Label className="text-2xs text-muted-foreground" title="When on, preprocessors run fresh every generation. When off, manually processed images are used as-is.">Re-process on generate</Label>
        <Switch checked={reprocessOnGenerate} onCheckedChange={setAutoUpdateProcessed} />
      </div>

      <div className="flex flex-col gap-1.5">
        <CanvasInputRow />
        {units.map((_, i) => (
          <ControlUnitRow key={i} index={i} unifiedIndex={i + 2} canRemove={units.length > 1} />
        ))}
      </div>

      <AddInputPopover
        availableSubTypes={availableControlSubTypes}
        onAdd={addUnitWithType}
        disabled={units.length >= 10}
      />
    </div>
  );
}

interface ControlUnitRowProps {
  index: number;
  unifiedIndex: number;
  canRemove: boolean;
}

function ControlUnitRow({ index, unifiedIndex, canRemove }: ControlUnitRowProps) {
  const unit = useControlStore((s) => s.units[index]);
  const units = useControlStore((s) => s.units);
  const setUnitParam = useControlStore((s) => s.setUnitParam);
  const setUnitType = useControlStore((s) => s.setUnitType);
  const removeUnit = useControlStore((s) => s.removeUnit);
  const setImageSource = useControlStore((s) => s.setImageSource);
  const setSelectedControlFrame = useCanvasStore((s) => s.setSelectedControlFrame);

  const typeOptions = useMemo(() => {
    const otherLocked = units
      .filter((u, i) => i !== index && u.enabled && EXCLUSIVE_CONTROL_TYPES.has(u.unitType))
      .map((u) => u.unitType)[0] ?? null;

    return UNIT_TYPE_OPTIONS.map((opt) => ({
      ...opt,
      disabled: otherLocked !== null
        && EXCLUSIVE_CONTROL_TYPES.has(opt.value)
        && opt.value !== otherLocked,
    }));
  }, [units, index]);

  const imageSourceOptions = useMemo(() => {
    const opts: { value: string; label: string }[] = [
      { value: "canvas", label: "Input 1 image" },
      { value: "separate", label: "Own image" },
    ];
    units.forEach((u, i) => {
      if (i !== index && u.imageSource === "separate") {
        opts.push({ value: `unit:${i}`, label: `Input ${i + 2} (${UNIT_TYPE_LABELS[u.unitType] ?? u.unitType}) image` });
      }
    });
    return opts;
  }, [units, index]);

  const handleEditOnCanvas = useCallback(() => {
    const match = unit.imageSource.match(/^unit:(\d+)$/);
    const targetIndex = match ? Number(match[1]) : index;
    setSelectedControlFrame(targetIndex);
  }, [setSelectedControlFrame, index, unit.imageSource]);

  const showEditOnCanvas = unit.enabled && (unit.imageSource === "separate" || unit.imageSource.startsWith("unit:"));

  return (
    <div className="flex flex-col gap-1.5 p-2 rounded-md border border-border">
      {/* Row 1: Index + Type + Enabled + Remove */}
      <div className="flex items-center gap-1.5">
        <span className="text-2xs text-muted-foreground font-mono w-4 flex-shrink-0">{unifiedIndex}</span>
        <Combobox
          value={unit.unitType}
          onValueChange={(v) => setUnitType(index, v as ControlUnitType)}
          options={typeOptions}
          className="h-6 text-2xs flex-1"
        />
        <div className="flex items-center gap-1">
          <Label className="text-2xs text-muted-foreground">On</Label>
          <Switch checked={unit.enabled} onCheckedChange={(checked) => setUnitParam(index, "enabled", checked)} />
        </div>
        {canRemove && (
          <Button variant="ghost" size="icon-sm" onClick={() => removeUnit(index)} title="Remove input">
            <Trash2 size={12} />
          </Button>
        )}
      </div>

      {/* Row 2: Image source selector + Edit on Canvas */}
      <div className="flex items-center justify-between gap-2">
        <Combobox
          value={unit.imageSource}
          onValueChange={(v) => setImageSource(index, v)}
          options={imageSourceOptions}
          className="h-6 text-2xs flex-1"
        />
        {showEditOnCanvas && (
          <Button
            variant="link"
            size="sm"
            className="h-auto p-0 text-2xs text-amber-500 gap-1 shrink-0"
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
