import { useControlStore } from "@/stores/controlStore";
import { ControlUnitCard } from "./control/ControlUnitCard";
import { ParamSection } from "../ParamSection";
import { ParamSlider } from "../ParamSlider";
import { Button } from "@/components/ui/button";
import { Plus } from "lucide-react";

export function ControlTab() {
  const units = useControlStore((s) => s.units);
  const addUnit = useControlStore((s) => s.addUnit);
  const setUnitCount = useControlStore((s) => s.setUnitCount);

  return (
    <div className="flex flex-col gap-3 text-sm">
      <ParamSlider label="Units" value={units.length} onChange={setUnitCount} min={1} max={10} />

      <ParamSection title="Control Units">
        <div className="flex flex-col gap-3">
          {units.map((_, i) => (
            <ControlUnitCard key={i} index={i} canRemove={units.length > 1} />
          ))}
        </div>

        <Button variant="outline" size="sm" className="w-full mt-2" onClick={addUnit} disabled={units.length >= 10}>
          <Plus size={12} className="mr-1" /> Add Unit
        </Button>
      </ParamSection>
    </div>
  );
}
