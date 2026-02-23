import { useCallback, useRef } from "react";
import { Label } from "@/components/ui/label";

interface ColorPickerProps {
  label: string;
  value: string;
  onChange: (value: string) => void;
}

export function ColorPicker({ label, value, onChange }: ColorPickerProps) {
  const inputRef = useRef<HTMLInputElement>(null);

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      onChange(e.target.value);
    },
    [onChange],
  );

  return (
    <div className="flex items-center gap-2">
      <Label className="text-2xs text-muted-foreground w-16 flex-shrink-0">{label}</Label>
      <div className="flex items-center gap-1.5 flex-1">
        <input
          ref={inputRef}
          type="color"
          value={value}
          onChange={handleChange}
          className="h-6 w-8 cursor-pointer rounded border border-border bg-transparent p-0"
        />
        <span className="text-2xs text-muted-foreground font-mono">{value}</span>
      </div>
    </div>
  );
}
