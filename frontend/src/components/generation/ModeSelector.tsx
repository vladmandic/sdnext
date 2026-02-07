import { useUiStore } from "@/stores/uiStore";
import type { GenerationMode } from "@/stores/uiStore";
import { cn } from "@/lib/utils";

const modes: { value: GenerationMode; label: string }[] = [
  { value: "txt2img", label: "txt2img" },
  { value: "img2img", label: "img2img" },
];

export function ModeSelector() {
  const generationMode = useUiStore((s) => s.generationMode);
  const setGenerationMode = useUiStore((s) => s.setGenerationMode);

  return (
    <div className="flex h-7 rounded-md border border-border overflow-hidden">
      {modes.map((mode) => (
        <button
          key={mode.value}
          type="button"
          onClick={() => setGenerationMode(mode.value)}
          className={cn(
            "flex-1 text-[11px] font-medium transition-colors",
            generationMode === mode.value
              ? "bg-primary text-primary-foreground"
              : "bg-muted text-muted-foreground hover:text-foreground",
          )}
        >
          {mode.label}
        </button>
      ))}
    </div>
  );
}
