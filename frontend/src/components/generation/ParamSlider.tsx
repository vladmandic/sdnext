import { memo, useCallback } from "react";
import { Slider } from "@/components/ui/slider";
import { NumberInput } from "@/components/ui/number-input";
import { ParamLabel } from "./ParamLabel";

interface ParamSliderProps {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step?: number;
  disabled?: boolean;
  tooltip?: string;
}

export const ParamSlider = memo(function ParamSlider({ label, value, onChange, min, max, step = 1, disabled, tooltip }: ParamSliderProps) {
  const handleSliderChange = useCallback(([v]: number[]) => onChange(v), [onChange]);

  return (
    <div data-param={label.toLowerCase()} className={disabled ? "opacity-50 pointer-events-none" : undefined}>
      <div className="flex items-center justify-between mb-0.5">
        <ParamLabel className="text-2xs text-muted-foreground" tooltip={tooltip}>{label}</ParamLabel>
        <NumberInput
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={onChange}
          fallback={min}
          className="w-12 h-5 text-2xs text-right px-1 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
          disabled={disabled}
        />
      </div>
      <Slider
        min={min}
        max={max}
        step={step}
        value={[value]}
        onValueChange={handleSliderChange}
        disabled={disabled}
      />
    </div>
  );
});
