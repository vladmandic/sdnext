import { memo, useCallback } from "react";
import { Slider } from "@/components/ui/slider";
import { NumberInput } from "@/components/ui/number-input";
import { Label } from "@/components/ui/label";

interface ParamSliderProps {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step?: number;
}

export const ParamSlider = memo(function ParamSlider({ label, value, onChange, min, max, step = 1 }: ParamSliderProps) {
  const handleSliderChange = useCallback(([v]: number[]) => onChange(v), [onChange]);

  return (
    <div className="flex items-center gap-2">
      <Label className="text-[11px] text-muted-foreground w-16 flex-shrink-0">{label}</Label>
      <Slider
        min={min}
        max={max}
        step={step}
        value={[value]}
        onValueChange={handleSliderChange}
        className="flex-1"
      />
      <NumberInput
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={onChange}
        fallback={min}
        className="w-14 h-6 text-[11px] text-center px-1 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
      />
    </div>
  );
});
