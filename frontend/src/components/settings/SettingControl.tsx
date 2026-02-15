import type { SettingDef } from "@/lib/settingsSchema";
import { cn } from "@/lib/utils";
import { Switch } from "@/components/ui/switch";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { NumberInput } from "@/components/ui/number-input";
import { Slider } from "@/components/ui/slider";
import { Combobox } from "@/components/ui/combobox";

function renderSelect(choices: string[] | undefined, value: unknown, onChange: (value: unknown) => void, setting: SettingDef) {
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
    <Combobox
      value={String(value ?? "")}
      onValueChange={(v) => onChange(v)}
      options={choices}
      placeholder="Select..."
      className="h-7 text-xs min-w-[140px]"
    />
  );
}

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

    case "slider": {
      const numVal = typeof value === "number" ? value : (setting.defaultValue as number) ?? 0;
      const display = setting.precision != null ? numVal.toFixed(setting.precision) : String(numVal);
      return (
        <div className="flex items-center gap-2 flex-1">
          <Slider
            min={setting.min ?? 0}
            max={setting.max ?? 100}
            step={setting.step ?? 1}
            value={[numVal]}
            onValueChange={([v]) => onChange(v)}
            className="flex-1"
          />
          <span className="text-xs text-muted-foreground tabular-nums w-14 text-right">
            {display}
          </span>
        </div>
      );
    }

    case "radio":
      if (choices && choices.length > 0 && choices.length <= 5) {
        return (
          <div className="inline-flex border border-border bg-muted/40 p-0.5" style={{ borderRadius: "var(--control-radius)" }}>
            {choices.map((choice) => (
              <button
                key={choice}
                type="button"
                onClick={() => onChange(choice)}
                className={cn(
                  "px-2.5 py-1 text-xs transition-colors",
                  String(value) === choice
                    ? "bg-background text-foreground shadow-sm"
                    : "text-muted-foreground hover:text-foreground",
                )}
                style={{ borderRadius: "var(--control-inner-radius)" }}
              >
                {choice}
              </button>
            ))}
          </div>
        );
      }
      return renderSelect(choices, value, onChange, setting);

    case "select":
      return renderSelect(choices, value, onChange, setting);

    case "multiselect": {
      const selected = Array.isArray(value) ? value as string[] : [];
      if (!choices || choices.length === 0) {
        return (
          <Input
            value={selected.join(", ")}
            onChange={(e) => onChange(e.target.value.split(",").map((s) => s.trim()).filter(Boolean))}
            className="h-7 text-xs"
            placeholder="Comma-separated values"
          />
        );
      }
      return (
        <div className="flex flex-wrap gap-x-3 gap-y-1.5">
          {choices.map((choice) => {
            const checked = selected.includes(choice);
            return (
              <label key={choice} className="flex items-center gap-1.5 text-xs cursor-pointer">
                <Checkbox
                  checked={checked}
                  onCheckedChange={(c) => {
                    if (c) onChange([...selected, choice]);
                    else onChange(selected.filter((s) => s !== choice));
                  }}
                />
                {choice}
              </label>
            );
          })}
        </div>
      );
    }

    case "number": {
      const step = setting.step ?? (setting.precision != null ? 10 ** -setting.precision : undefined);
      return (
        <NumberInput
          min={setting.min}
          max={setting.max}
          step={step}
          value={typeof value === "number" ? value : 0}
          onChange={(v) => onChange(v)}
          fallback={0}
          className="h-7 text-xs w-24 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
        />
      );
    }

    case "color":
      return (
        <div className="flex items-center gap-2">
          <input
            type="color"
            value={String(value ?? "#000000")}
            onChange={(e) => onChange(e.target.value)}
            className="h-7 w-7 rounded border border-border cursor-pointer p-0.5"
          />
          <Input
            value={String(value ?? "")}
            onChange={(e) => onChange(e.target.value)}
            className="h-7 text-xs w-24"
            placeholder="#000000"
          />
        </div>
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
