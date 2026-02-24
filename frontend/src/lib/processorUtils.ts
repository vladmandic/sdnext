import type { PreprocessorInfo } from "@/api/types/control";
import type { ComboboxGroup } from "@/components/ui/combobox";

const GROUP_ORDER = ["Pose", "Edge", "Depth", "Normal", "Segmentation", "Other"] as const;

export function buildProcessorGroups(preprocessors: PreprocessorInfo[]): ComboboxGroup[] {
  const buckets: Record<string, string[]> = {};
  for (const p of preprocessors) {
    if (p.name === "None") continue;
    const group = p.group || "Other";
    (buckets[group] ??= []).push(p.name);
  }
  for (const names of Object.values(buckets)) {
    names.sort((a, b) => a.localeCompare(b));
  }
  return GROUP_ORDER
    .filter((g) => buckets[g]?.length)
    .map((g) => ({ heading: g, options: buckets[g] }));
}
