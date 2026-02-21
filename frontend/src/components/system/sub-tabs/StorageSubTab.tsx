import { useStorage } from "@/api/hooks/useSystem";
import { formatBytes } from "@/lib/utils";
import { Section, BarRow, Row } from "./OverviewSubTab";
import { Button } from "@/components/ui/button";
import { AlertCircle, HardDrive, Loader2 } from "lucide-react";

const CATEGORY_LABELS: Record<string, string> = {
  caches: "Caches",
  temp: "Temporary Files",
  huggingface: "HuggingFace Cache",
  outputs: "Outputs",
  models: "Models",
};

export function StorageSubTab() {
  const { data, isError, isFetching, refetch } = useStorage();

  return (
    <div className="space-y-4">
      <Button size="sm" variant="outline" onClick={() => refetch()} disabled={isFetching}>
        {isFetching ? (
          <Loader2 className="h-3.5 w-3.5 mr-1 animate-spin" />
        ) : (
          <HardDrive className="h-3.5 w-3.5 mr-1" />
        )}
        {isError ? "Retry" : isFetching ? "Scanning..." : "Scan Storage"}
      </Button>

      {isError && (
        <div className="flex items-center gap-2 rounded-md border border-destructive/30 bg-destructive/5 px-3 py-2">
          <AlertCircle className="h-4 w-4 shrink-0 text-destructive" />
          <span className="text-xs text-destructive">Could not reach backend — storage info unavailable</span>
        </div>
      )}

      {data && Object.entries(data).map(([category, entries]) => {
        const total = entries.reduce((sum, e) => sum + e.size, 0);
        if (total === 0 && entries.every((e) => e.size === 0)) return null;
        const label = CATEGORY_LABELS[category] ?? category;
        return (
          <Section key={category} title={`${label} (${formatBytes(total)})`}>
            {entries.map((entry) => (
              entry.size > 0 ? (
                <BarRow key={entry.path} label={entry.label} value={entry.size} max={total} formatter={formatBytes} />
              ) : (
                <Row key={entry.path} label={entry.label} value="0 B" />
              )
            ))}
          </Section>
        );
      })}
    </div>
  );
}
