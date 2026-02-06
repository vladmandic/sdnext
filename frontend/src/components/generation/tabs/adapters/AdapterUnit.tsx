import { useAdapterStore } from "@/stores/adapterStore";
import { useIPAdapterModels } from "@/api/hooks/useAdapters";
import { ParamSlider } from "../../ParamSlider";
import { ParamSection } from "../../ParamSection";
import { ImageUpload } from "../../ImageUpload";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { X } from "lucide-react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useMemo } from "react";

interface AdapterUnitProps {
  index: number;
}

export function AdapterUnit({ index }: AdapterUnitProps) {
  const unit = useAdapterStore((s) => s.units[index]);
  const setUnitParam = useAdapterStore((s) => s.setUnitParam);
  const addUnitImage = useAdapterStore((s) => s.addUnitImage);
  const removeUnitImage = useAdapterStore((s) => s.removeUnitImage);
  const addUnitMask = useAdapterStore((s) => s.addUnitMask);
  const removeUnitMask = useAdapterStore((s) => s.removeUnitMask);
  const { data: adapterModels } = useIPAdapterModels();

  const previews = useMemo(
    () => unit.images.map((f) => URL.createObjectURL(f)),
    [unit.images],
  );

  const maskPreviews = useMemo(
    () => unit.masks.map((f) => URL.createObjectURL(f)),
    [unit.masks],
  );

  return (
    <div className="flex flex-col gap-2 p-2 rounded-md border border-border">
      <div className="flex items-center gap-2">
        <Label className="text-[11px] text-muted-foreground w-16 flex-shrink-0">Adapter</Label>
        <Select value={unit.adapter} onValueChange={(v) => setUnitParam(index, "adapter", v)}>
          <SelectTrigger className="h-7 text-xs flex-1">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="None">None</SelectItem>
            {adapterModels?.map((name) => (
              <SelectItem key={name} value={name}>{name}</SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <ParamSlider label="Scale" value={unit.scale} onChange={(v) => setUnitParam(index, "scale", v)} min={0} max={2} step={0.01} />

      <div className="flex items-center gap-2">
        <Label className="text-[11px] text-muted-foreground w-16 flex-shrink-0">Crop</Label>
        <Switch checked={unit.crop} onCheckedChange={(checked) => setUnitParam(index, "crop", checked)} />
      </div>

      <ParamSection title="Timing" defaultOpen={false}>
        <ParamSlider label="Start" value={unit.start} onChange={(v) => setUnitParam(index, "start", v)} min={0} max={1} step={0.01} />
        <ParamSlider label="End" value={unit.end} onChange={(v) => setUnitParam(index, "end", v)} min={0} max={1} step={0.01} />
      </ParamSection>

      <div className="flex flex-col gap-1">
        <Label className="text-[11px] text-muted-foreground">Images</Label>
        <div className="flex gap-1 flex-wrap">
          {previews.map((url, i) => (
            <div key={i} className="relative h-16 w-16 rounded border border-border overflow-hidden group">
              <img src={url} alt={`ref ${i}`} className="w-full h-full object-cover" />
              <Button
                variant="destructive"
                size="icon-sm"
                className="absolute top-0 right-0 opacity-0 group-hover:opacity-100 h-4 w-4"
                onClick={() => removeUnitImage(index, i)}
              >
                <X size={8} />
              </Button>
            </div>
          ))}
        </div>
        <ImageUpload
          image={null}
          onImageChange={(file) => { if (file) addUnitImage(index, file); }}
          label="Add reference"
          compact
        />
      </div>

      <div className="flex flex-col gap-1">
        <Label className="text-[11px] text-muted-foreground">Masks</Label>
        <div className="flex gap-1 flex-wrap">
          {maskPreviews.map((url, i) => (
            <div key={i} className="relative h-16 w-16 rounded border border-border overflow-hidden group">
              <img src={url} alt={`mask ${i}`} className="w-full h-full object-cover" />
              <Button
                variant="destructive"
                size="icon-sm"
                className="absolute top-0 right-0 opacity-0 group-hover:opacity-100 h-4 w-4"
                onClick={() => removeUnitMask(index, i)}
              >
                <X size={8} />
              </Button>
            </div>
          ))}
        </div>
        <ImageUpload
          image={null}
          onImageChange={(file) => { if (file) addUnitMask(index, file); }}
          label="Add mask"
          compact
        />
      </div>
    </div>
  );
}
