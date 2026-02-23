import { useState, useMemo } from "react";
import { Loader2 } from "lucide-react";
import { useModelListDetail, useUpdateHashes } from "@/api/hooks/useModelOps";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

function formatSize(bytes: number): string {
  if (bytes <= 0) return "-";
  const gb = bytes / (1024 * 1024 * 1024);
  if (gb >= 1) return `${gb.toFixed(2)} GB`;
  const mb = bytes / (1024 * 1024);
  return `${mb.toFixed(1)} MB`;
}

export function ListSubTab() {
  const { data: models, isLoading } = useModelListDetail();
  const updateHashes = useUpdateHashes();
  const [search, setSearch] = useState("");

  const filtered = useMemo(() => {
    if (!models) return [];
    if (!search) return models;
    const q = search.toLowerCase();
    return models.filter(
      (m) =>
        m.model_name.toLowerCase().includes(q) ||
        m.detected_type.toLowerCase().includes(q) ||
        (m.pipeline ?? "").toLowerCase().includes(q),
    );
  }, [models, search]);

  return (
    <div className="space-y-2">
      <div className="flex gap-2">
        <Input placeholder="Filter..." value={search} onChange={(e) => setSearch(e.target.value)} className="h-6 text-2xs flex-1" />
        <Button
          size="sm"
          variant="secondary"
          onClick={() => updateHashes.mutate()}
          disabled={updateHashes.isPending}
          className="shrink-0 text-2xs"
        >
          {updateHashes.isPending && <Loader2 className="h-3 w-3 animate-spin mr-1" />}
          Calc hashes
        </Button>
      </div>

      {isLoading && <p className="text-xs text-muted-foreground">Loading model list...</p>}

      {!isLoading && filtered.length > 0 && (
        <div className="border border-border rounded-md overflow-auto">
          <table className="w-full text-2xs">
            <thead>
              <tr className="bg-muted/50 border-b border-border">
                <th className="px-2 py-1 text-left font-medium">Name</th>
                <th className="px-2 py-1 text-left font-medium">Type</th>
                <th className="px-2 py-1 text-left font-medium">Detect</th>
                <th className="px-2 py-1 text-left font-medium">Pipeline</th>
                <th className="px-2 py-1 text-left font-medium">Hash</th>
                <th className="px-2 py-1 text-right font-medium">Size</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((m) => (
                <tr key={m.filename} className="border-b border-border/50 hover:bg-muted/30">
                  <td className="px-2 py-1 truncate max-w-[8.75rem] font-mono" title={m.filename}>{m.model_name}</td>
                  <td className="px-2 py-1">{m.type}</td>
                  <td className="px-2 py-1">{m.detected_type}</td>
                  <td className="px-2 py-1 truncate max-w-25">{m.pipeline ?? "-"}</td>
                  <td className="px-2 py-1 font-mono">{m.hash ?? "-"}</td>
                  <td className="px-2 py-1 text-right font-mono">{formatSize(m.size)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {!isLoading && filtered.length === 0 && models && models.length > 0 && (
        <p className="text-xs text-muted-foreground">No models match filter.</p>
      )}

      {!isLoading && (!models || models.length === 0) && (
        <p className="text-xs text-muted-foreground">No models found.</p>
      )}

      <p className="text-3xs text-muted-foreground">{models?.length ?? 0} models total</p>
    </div>
  );
}
