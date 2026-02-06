import { useControlStore } from "@/stores/controlStore";
import { ControlUnitCard } from "./control/ControlUnitCard";
import { ParamSection } from "../ParamSection";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Plus } from "lucide-react";
import type { ControlUnitType } from "@/api/types/control";

const UNIT_TYPES: { value: ControlUnitType; label: string; available: boolean }[] = [
  { value: "controlnet", label: "ControlNet", available: true },
  { value: "t2i", label: "T2I-Adapter", available: false },
  { value: "xs", label: "XS", available: false },
  { value: "lite", label: "Lite", available: false },
  { value: "reference", label: "Reference", available: false },
];

export function ControlTab() {
  const activeType = useControlStore((s) => s.activeType);
  const units = useControlStore((s) => s.units);
  const addUnit = useControlStore((s) => s.addUnit);
  const setActiveType = useControlStore((s) => s.setActiveType);
  const guessMode = useControlStore((s) => s.guessMode);

  return (
    <div className="flex flex-col gap-3 text-sm">
      {/* Type tabs */}
      <div className="flex gap-1">
        {UNIT_TYPES.map((t) => (
          <button
            key={t.value}
            onClick={() => t.available && setActiveType(t.value)}
            className={`px-2 py-1 text-[10px] rounded transition-colors ${activeType === t.value ? "bg-primary text-primary-foreground" : t.available ? "bg-muted text-muted-foreground hover:text-foreground" : "bg-muted/50 text-muted-foreground/50 cursor-not-allowed"}`}
            disabled={!t.available}
            title={t.available ? t.label : `${t.label} — Coming soon`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {activeType === "controlnet" ? (
        <>
          <label className="flex items-center gap-1.5 text-[11px] text-muted-foreground cursor-pointer px-1">
            <Checkbox checked={guessMode} onCheckedChange={(c) => useControlStore.setState({ guessMode: !!c })} />
            Guess mode
          </label>

          <ParamSection title="ControlNet Units">
            <div className="flex flex-col gap-3">
              {units.map((_, i) => (
                <ControlUnitCard key={i} index={i} canRemove={units.length > 1} />
              ))}
            </div>

            <Button variant="outline" size="sm" className="w-full mt-2" onClick={addUnit}>
              <Plus size={12} className="mr-1" /> Add Unit
            </Button>
          </ParamSection>
        </>
      ) : (
        <div className="flex flex-col items-center justify-center h-32 text-muted-foreground">
          <p className="text-xs">Coming soon</p>
        </div>
      )}
    </div>
  );
}
