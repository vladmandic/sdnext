import { useState } from "react";
import { Loader2, ChevronDown, ChevronRight, ExternalLink } from "lucide-react";
import { useHfSearch, useHfDownload } from "@/api/hooks/useModelOps";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

function formatDownloads(n: number): string {
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return String(n);
}

export function HuggingfaceSubTab() {
  const [keyword, setKeyword] = useState("");
  const [searchEnabled, setSearchEnabled] = useState(false);
  const { data: results, isLoading, refetch } = useHfSearch(keyword, searchEnabled);
  const hfDownload = useHfDownload();
  const [selected, setSelected] = useState<string | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [token, setToken] = useState("");
  const [variant, setVariant] = useState("");
  const [revision, setRevision] = useState("");
  const [mirror, setMirror] = useState("");
  const [customPipeline, setCustomPipeline] = useState("");

  function handleSearch() {
    if (!keyword) return;
    if (searchEnabled) {
      refetch();
    } else {
      setSearchEnabled(true);
    }
  }

  function handleDownload() {
    if (!selected) return;
    hfDownload.mutate({
      hub_id: selected,
      token: token || undefined,
      variant: variant || undefined,
      revision: revision || undefined,
      mirror: mirror || undefined,
      custom_pipeline: customPipeline || undefined,
    });
  }

  return (
    <div className="space-y-3">
      <div className="flex gap-2">
        <Input
          placeholder="Search HuggingFace..."
          value={keyword}
          onChange={(e) => { setKeyword(e.target.value); setSearchEnabled(false); }}
          onKeyDown={(e) => e.key === "Enter" && handleSearch()}
          className="h-7 text-xs flex-1"
        />
        <Button size="sm" variant="secondary" onClick={handleSearch} disabled={isLoading || !keyword} className="shrink-0">
          {isLoading && <Loader2 className="h-3 w-3 animate-spin mr-1" />}
          Search
        </Button>
      </div>

      {results && results.length > 0 && (
        <div className="border border-border rounded-md overflow-auto max-h-[300px]">
          <table className="w-full text-[11px]">
            <thead>
              <tr className="bg-muted/50 border-b border-border">
                <th className="px-2 py-1 text-left font-medium">Name</th>
                <th className="px-2 py-1 text-left font-medium">Pipeline</th>
                <th className="px-2 py-1 text-right font-medium">Downloads</th>
                <th className="px-2 py-1 text-center font-medium w-6"></th>
              </tr>
            </thead>
            <tbody>
              {results.map((r) => (
                <tr
                  key={r.id}
                  onClick={() => setSelected(r.id)}
                  className={`border-b border-border/50 cursor-pointer ${selected === r.id ? "bg-accent/30" : "hover:bg-muted/30"}`}
                >
                  <td className="px-2 py-1 truncate max-w-[160px] font-mono">{r.id}</td>
                  <td className="px-2 py-1">{r.pipeline_tag ?? "-"}</td>
                  <td className="px-2 py-1 text-right font-mono">{formatDownloads(r.downloads)}</td>
                  <td className="px-2 py-1 text-center">
                    {r.url && (
                      <a href={r.url} target="_blank" rel="noopener noreferrer" onClick={(e) => e.stopPropagation()} className="text-muted-foreground hover:text-foreground">
                        <ExternalLink className="h-3 w-3" />
                      </a>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {selected && (
        <div className="space-y-2 border border-border rounded-md p-2">
          <p className="text-xs font-medium font-mono">{selected}</p>

          <button type="button" onClick={() => setShowAdvanced(!showAdvanced)} className="flex items-center gap-1 text-[11px] text-muted-foreground hover:text-foreground">
            {showAdvanced ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
            Advanced options
          </button>
          {showAdvanced && (
            <div className="space-y-2 pl-3">
              <div>
                <Label className="text-[11px]">Token</Label>
                <Input className="h-7 text-xs" type="password" value={token} onChange={(e) => setToken(e.target.value)} placeholder="HuggingFace token" />
              </div>
              <div>
                <Label className="text-[11px]">Variant</Label>
                <Input className="h-7 text-xs" value={variant} onChange={(e) => setVariant(e.target.value)} placeholder="e.g. fp16" />
              </div>
              <div>
                <Label className="text-[11px]">Revision</Label>
                <Input className="h-7 text-xs" value={revision} onChange={(e) => setRevision(e.target.value)} placeholder="branch or commit" />
              </div>
              <div>
                <Label className="text-[11px]">Mirror</Label>
                <Input className="h-7 text-xs" value={mirror} onChange={(e) => setMirror(e.target.value)} placeholder="optional mirror URL" />
              </div>
              <div>
                <Label className="text-[11px]">Custom pipeline</Label>
                <Input className="h-7 text-xs" value={customPipeline} onChange={(e) => setCustomPipeline(e.target.value)} placeholder="pipeline name" />
              </div>
            </div>
          )}

          <Button size="sm" variant="secondary" onClick={handleDownload} disabled={hfDownload.isPending} className="w-full">
            {hfDownload.isPending && <Loader2 className="h-3 w-3 animate-spin mr-1" />}
            Download
          </Button>
          {hfDownload.data && <p className="text-[11px] text-muted-foreground">{hfDownload.data.status}</p>}
        </div>
      )}
    </div>
  );
}
