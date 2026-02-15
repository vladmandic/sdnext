import { Loader2 } from "lucide-react";
import { useMetadataScan, useMetadataUpdate } from "@/api/hooks/useModelOps";
import { Button } from "@/components/ui/button";

export function MetadataSubTab() {
  const scan = useMetadataScan();
  const update = useMetadataUpdate();

  return (
    <div className="space-y-3">
      <p className="text-xs text-muted-foreground">Fetch model preview metadata from CivitAI.</p>

      <div className="flex gap-2">
        <Button size="sm" variant="secondary" onClick={() => scan.mutate()} disabled={scan.isPending} className="flex-1">
          {scan.isPending && <Loader2 className="h-3 w-3 animate-spin mr-1" />}
          Scan missing
        </Button>
        <Button size="sm" variant="secondary" onClick={() => update.mutate()} disabled={update.isPending} className="flex-1">
          {update.isPending && <Loader2 className="h-3 w-3 animate-spin mr-1" />}
          Update all
        </Button>
      </div>

      {scan.data && scan.data.results.length > 0 && (
        <div className="space-y-1">
          <p className="text-xs font-medium">Scan results ({scan.data.results.length})</p>
          <div className="border border-border rounded-md overflow-auto max-h-[300px]">
            <table className="w-full text-[11px]">
              <thead>
                <tr className="bg-muted/50 border-b border-border">
                  <th className="px-2 py-1 text-left font-medium">Name</th>
                  <th className="px-2 py-1 text-left font-medium">Type</th>
                  <th className="px-2 py-1 text-left font-medium">Hash</th>
                  <th className="px-2 py-1 text-left font-medium">Note</th>
                </tr>
              </thead>
              <tbody>
                {scan.data.results.map((r, i) => (
                  <tr key={i} className="border-b border-border/50 hover:bg-muted/30">
                    <td className="px-2 py-1 truncate max-w-[120px]">{r.name}</td>
                    <td className="px-2 py-1">{r.type}</td>
                    <td className="px-2 py-1 font-mono">{r.hash?.slice(0, 10) ?? "-"}</td>
                    <td className="px-2 py-1 truncate max-w-[100px]">{r.note}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {update.data && update.data.results.length > 0 && (
        <div className="space-y-1">
          <p className="text-xs font-medium">Update results ({update.data.results.length})</p>
          <div className="border border-border rounded-md overflow-auto max-h-[300px]">
            <table className="w-full text-[11px]">
              <thead>
                <tr className="bg-muted/50 border-b border-border">
                  <th className="px-2 py-1 text-left font-medium">Name</th>
                  <th className="px-2 py-1 text-left font-medium">Versions</th>
                  <th className="px-2 py-1 text-left font-medium">Latest</th>
                  <th className="px-2 py-1 text-left font-medium">Status</th>
                </tr>
              </thead>
              <tbody>
                {update.data.results.map((r, i) => (
                  <tr key={i} className="border-b border-border/50 hover:bg-muted/30">
                    <td className="px-2 py-1 truncate max-w-[120px]">{r.name ?? r.file}</td>
                    <td className="px-2 py-1">{r.versions ?? "-"}</td>
                    <td className="px-2 py-1 truncate max-w-[80px]">{r.latest ?? "-"}</td>
                    <td className="px-2 py-1">{r.status ?? "-"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
