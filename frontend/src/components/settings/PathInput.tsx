import { useMemo } from "react";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";

function isAbsolutePath(p: string): boolean {
  return p.startsWith("/") || /^[A-Za-z]:[\\/]/.test(p);
}

function resolvePreview(base: string, specific: string): string {
  if (!specific) return base || "";
  if (isAbsolutePath(specific)) return specific;
  if (base) return `${base.replace(/\/+$/, "")}/${specific}`.replace(/\/+/g, "/");
  return specific;
}

interface PathInputProps {
  value: string;
  onChange: (value: string) => void;
  basePath: string;
  placeholder?: string;
}

export function PathInput({ value, onChange, basePath, placeholder }: PathInputProps) {
  const showPrefix = basePath.length > 0 && !isAbsolutePath(value);
  const preview = useMemo(() => resolvePreview(basePath, value), [basePath, value]);
  const prefixLabel = basePath.replace(/\/+$/, "") + "/";

  return (
    <div className="flex flex-col gap-0.5">
      <div className="flex items-stretch">
        {showPrefix && (
          <span className={cn(
            "inline-flex items-center px-2 text-2xs font-mono",
            "bg-muted/50 text-muted-foreground border border-r-0 border-input rounded-l-md",
            "select-none whitespace-nowrap",
          )}>
            {prefixLabel}
          </span>
        )}
        <Input
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          className={cn("h-7 text-2xs font-mono", showPrefix && "rounded-l-none border-l-0")}
        />
      </div>
      {!showPrefix && preview && (
        <span className="text-3xs text-muted-foreground font-mono truncate">
          Saves to: {preview}
        </span>
      )}
    </div>
  );
}
