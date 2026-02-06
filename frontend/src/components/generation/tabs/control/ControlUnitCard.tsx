import { useControlStore } from "@/stores/controlStore";
import { useControlNetModels, usePreprocessors } from "@/api/hooks/useControl";
import { ParamSlider } from "../../ParamSlider";
import { ParamSection } from "../../ParamSection";
import { ImageUpload } from "../../ImageUpload";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Trash2 } from "lucide-react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

interface ControlUnitCardProps {
  index: number;
  canRemove: boolean;
}

export function ControlUnitCard({ index, canRemove }: ControlUnitCardProps) {
  const unit = useControlStore((s) => s.units[index]);
  const setUnitParam = useControlStore((s) => s.setUnitParam);
  const setUnitImage = useControlStore((s) => s.setUnitImage);
  const removeUnit = useControlStore((s) => s.removeUnit);
  const { data: models } = useControlNetModels();
  const { data: preprocessors } = usePreprocessors();

  return (
    <div className="flex flex-col gap-2 p-2 rounded-md border border-border">
      <div className="flex items-center justify-between">
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

      <div className="flex items-center gap-2">
        <Label className="text-[11px] text-muted-foreground w-16 flex-shrink-0">Processor</Label>
        <Select value={unit.processor} onValueChange={(v) => setUnitParam(index, "processor", v)}>
          <SelectTrigger className="h-7 text-xs flex-1">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="None">None</SelectItem>
            {preprocessors?.filter((p) => p.name !== "None").map((p) => (
              <SelectItem key={p.name} value={p.name}>{p.name}</SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <div className="flex items-center gap-2">
        <Label className="text-[11px] text-muted-foreground w-16 flex-shrink-0">Model</Label>
        <Select value={unit.model} onValueChange={(v) => setUnitParam(index, "model", v)}>
          <SelectTrigger className="h-7 text-xs flex-1">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="None">None</SelectItem>
            {models?.map((name) => (
              <SelectItem key={name} value={name}>{name}</SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <ParamSlider label="Strength" value={unit.strength} onChange={(v) => setUnitParam(index, "strength", v)} min={0.01} max={2} step={0.01} />

      <ParamSection title="Timing" defaultOpen={false}>
        <ParamSlider label="Start" value={unit.start} onChange={(v) => setUnitParam(index, "start", v)} min={0} max={1} step={0.01} />
        <ParamSlider label="End" value={unit.end} onChange={(v) => setUnitParam(index, "end", v)} min={0} max={1} step={0.01} />
      </ParamSection>

      <div className="flex flex-col gap-1">
        <Label className="text-[11px] text-muted-foreground">Control Image</Label>
        <ImageUpload
          image={unit.image}
          onImageChange={(file) => setUnitImage(index, file)}
          label="Drop control image"
          compact
        />
      </div>
    </div>
  );
}
