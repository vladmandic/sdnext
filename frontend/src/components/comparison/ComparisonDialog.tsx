import { useComparisonStore } from "@/stores/comparisonStore";
import { ComparisonView } from "./ComparisonView";

export function ComparisonDialog() {
  const open = useComparisonStore((s) => s.open);
  if (!open) return null;
  return <ComparisonView />;
}
