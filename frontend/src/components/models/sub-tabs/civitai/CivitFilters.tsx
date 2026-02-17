import type { CivitOptions } from "@/api/types/civitai";
import { Combobox } from "@/components/ui/combobox";
import { Label } from "@/components/ui/label";

interface CivitFiltersProps {
  options: CivitOptions | undefined;
  type: string;
  sort: string;
  period: string;
  baseModel: string;
  onTypeChange: (v: string) => void;
  onSortChange: (v: string) => void;
  onPeriodChange: (v: string) => void;
  onBaseModelChange: (v: string) => void;
}

export function CivitFilters({ options, type, sort, period, baseModel, onTypeChange, onSortChange, onPeriodChange, onBaseModelChange }: CivitFiltersProps) {
  const types = ["", ...(options?.types ?? [])];
  const sorts = ["", ...(options?.sort ?? [])];
  const periods = ["", ...(options?.period ?? [])];
  const baseModels = ["", ...(options?.base_models ?? [])];

  return (
    <div className="grid grid-cols-2 gap-2">
      <div>
        <Label className="text-[11px]">Type</Label>
        <Combobox value={type} onValueChange={onTypeChange} options={types} placeholder="All types" className="h-7 text-xs" />
      </div>
      <div>
        <Label className="text-[11px]">Sort</Label>
        <Combobox value={sort} onValueChange={onSortChange} options={sorts} placeholder="Default" className="h-7 text-xs" />
      </div>
      <div>
        <Label className="text-[11px]">Period</Label>
        <Combobox value={period} onValueChange={onPeriodChange} options={periods} placeholder="All time" className="h-7 text-xs" />
      </div>
      <div>
        <Label className="text-[11px]">Base model</Label>
        <Combobox value={baseModel} onValueChange={onBaseModelChange} options={baseModels} placeholder="Any" className="h-7 text-xs" />
      </div>
    </div>
  );
}
