import * as React from "react";
import { memo, useState, useEffect } from "react";
import { Input } from "@/components/ui/input";

interface NumberInputProps extends Omit<React.ComponentProps<"input">, "value" | "onChange" | "type"> {
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  fallback?: number;
}

/**
 * Number input that defers clamping until blur/Enter so users can freely
 * type multi-digit values without intermediate states being clamped.
 */
const NumberInput = memo(function NumberInput({ value, onChange, min, max, step, fallback, className, ...props }: NumberInputProps) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(String(value));

  useEffect(() => {
    if (!editing) setDraft(String(value));
  }, [value, editing]);

  const commit = () => {
    setEditing(false);
    const parsed = parseFloat(draft);
    if (Number.isNaN(parsed)) { onChange(fallback ?? min ?? 0); return; }
    let clamped = parsed;
    if (min != null) clamped = Math.max(min, clamped);
    if (max != null) clamped = Math.min(max, clamped);
    onChange(clamped);
  };

  return (
    <Input
      type="number"
      min={min}
      max={max}
      step={step}
      value={editing ? draft : value}
      onFocus={() => { setEditing(true); setDraft(String(value)); }}
      onChange={(e) => setDraft(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => { if (e.key === "Enter") commit(); }}
      className={className}
      {...props}
    />
  );
});

export { NumberInput };
