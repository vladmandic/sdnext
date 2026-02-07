import type { SettingDef } from "@/lib/settingsSchema";
import { Switch } from "@/components/ui/switch";
import { Input } from "@/components/ui/input";
import { NumberInput } from "@/components/ui/number-input";
import { Slider } from "@/components/ui/slider";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

interface SettingControlProps {
  setting: SettingDef;
  value: unknown;
  onChange: (value: unknown) => void;
  dynamicChoices?: string[];
}

export function SettingControl({ setting, value, onChange, dynamicChoices }: SettingControlProps) {
  const choices = dynamicChoices ?? setting.choices;

  switch (setting.component) {
    case "switch":
      return (
        <Switch
          checked={Boolean(value)}
          onCheckedChange={(checked) => onChange(checked)}
        />
      );

    case "slider":
      return (
        <div className="flex items-center gap-2 flex-1">
          <Slider
            min={setting.min ?? 0}
            max={setting.max ?? 100}
            step={setting.step ?? 1}
            value={[typeof value === "number" ? value : (setting.defaultValue as number) ?? 0]}
            onValueChange={([v]) => onChange(v)}
            className="flex-1"
          />
          <span className="text-xs text-muted-foreground tabular-nums w-10 text-right">
            {typeof value === "number" ? value : String(setting.defaultValue ?? "")}
          </span>
        </div>
      );

    case "select":
      if (!choices || choices.length === 0) {
        return (
          <Input
            value={String(value ?? "")}
            onChange={(e) => onChange(e.target.value)}
            className="h-7 text-xs"
            placeholder={setting.description}
          />
        );
      }
      return (
        <Select value={String(value ?? "")} onValueChange={(v) => onChange(v)}>
          <SelectTrigger size="sm" className="h-7 text-xs min-w-[140px]">
            <SelectValue placeholder="Select..." />
          </SelectTrigger>
          <SelectContent>
            {choices.map((c) => (
              <SelectItem key={c} value={c} className="text-xs">{c}</SelectItem>
            ))}
          </SelectContent>
        </Select>
      );

    case "number":
      return (
        <NumberInput
          min={setting.min}
          max={setting.max}
          step={setting.step}
          value={typeof value === "number" ? value : 0}
          onChange={(v) => onChange(v)}
          fallback={0}
          className="h-7 text-xs w-24 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
        />
      );

    case "input":
    default:
      return (
        <Input
          value={String(value ?? "")}
          onChange={(e) => onChange(e.target.value)}
          className="h-7 text-xs"
          placeholder={setting.description}
        />
      );
  }
}
