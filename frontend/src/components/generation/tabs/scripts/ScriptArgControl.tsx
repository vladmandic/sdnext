import { ParamSlider } from "../../ParamSlider";
import { Switch } from "@/components/ui/switch";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Combobox } from "@/components/ui/combobox";
import type { ScriptArg } from "@/api/types/script";

interface ScriptArgControlProps {
  arg: ScriptArg;
  value: unknown;
  onChange: (value: unknown) => void;
}

export function ScriptArgControl({ arg, value, onChange }: ScriptArgControlProps) {
  if (arg.choices && arg.choices.length > 0) {
    return (
      <div className="flex items-center gap-2">
        <Label className="text-[11px] text-muted-foreground w-20 flex-shrink-0 truncate" title={arg.label}>{arg.label}</Label>
        <Combobox
          value={String(value ?? arg.choices[0])}
          onValueChange={(v) => onChange(v)}
          options={arg.choices}
          className="h-7 text-xs flex-1"
        />
      </div>
    );
  }

  if (arg.minimum != null && arg.maximum != null && typeof (value ?? arg.value) === "number") {
    return (
      <ParamSlider
        label={arg.label}
        value={(value as number) ?? (arg.value as number) ?? arg.minimum}
        onChange={(v) => onChange(v)}
        min={arg.minimum}
        max={arg.maximum}
        step={arg.step ?? 1}
      />
    );
  }

  if (typeof (value ?? arg.value) === "boolean") {
    return (
      <div className="flex items-center gap-2">
        <Label className="text-[11px] text-muted-foreground w-20 flex-shrink-0 truncate" title={arg.label}>{arg.label}</Label>
        <Switch checked={Boolean(value ?? arg.value)} onCheckedChange={(checked) => onChange(checked)} />
      </div>
    );
  }

  return (
    <div className="flex items-center gap-2">
      <Label className="text-[11px] text-muted-foreground w-20 flex-shrink-0 truncate" title={arg.label}>{arg.label}</Label>
      <Input
        value={String(value ?? arg.value ?? "")}
        onChange={(e) => onChange(e.target.value)}
        className="h-7 text-xs flex-1"
      />
    </div>
  );
}
