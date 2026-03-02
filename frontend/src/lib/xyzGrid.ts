import type { ComboboxGroup } from "@/components/ui/combobox";
import type { XyzAxisOption } from "@/api/hooks/useXyzAxisOptions";

const RANGE_RE = /^(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)\s*:\s*(\d+)$/;

/** Parses a values string and returns the count of discrete values (client-side, no round-trip). */
export function countAxisValues(values: string, type: string): number {
  const trimmed = values.trim();
  if (!trimmed) return 0;

  const parts = trimmed.split(",").map((s) => s.trim()).filter(Boolean);
  if (parts.length === 0) return 0;

  if (type === "int" || type === "float") {
    let total = 0;
    for (const part of parts) {
      const m = RANGE_RE.exec(part);
      if (m) {
        total += Math.max(2, parseInt(m[3], 10));
      } else {
        total += 1;
      }
    }
    return total;
  }

  return parts.length;
}

/** Groups axis options by their `category` field from the v2 API. */
export function groupAxisOptions(options: XyzAxisOption[]): ComboboxGroup[] {
  const categoryMap = new Map<string, string[]>();

  for (const opt of options) {
    if (opt.label === "Nothing") continue;
    const cat = opt.category || "Other";
    if (!categoryMap.has(cat)) categoryMap.set(cat, []);
    categoryMap.get(cat)!.push(opt.label);
  }

  return Array.from(categoryMap.entries())
    .map(([heading, opts]) => ({ heading, options: opts }))
    .filter((g) => g.options.length > 0);
}
