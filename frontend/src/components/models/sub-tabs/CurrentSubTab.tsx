import { useState } from "react";
import { Loader2, ChevronDown, ChevronRight } from "lucide-react";
import { useModelAnalysis, useSaveModel } from "@/api/hooks/useModelOps";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";

function formatParams(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return String(n);
}

export function CurrentSubTab() {
  const [analyzeEnabled, setAnalyzeEnabled] = useState(false);
  const { data: analysis, isLoading, refetch } = useModelAnalysis(analyzeEnabled);
  const saveModel = useSaveModel();
  const [showSave, setShowSave] = useState(false);
  const [showMeta, setShowMeta] = useState(false);
  const [saveName, setSaveName] = useState("");
  const [savePath, setSavePath] = useState("");
  const [saveShard, setSaveShard] = useState("10GB");
  const [saveOverwrite, setSaveOverwrite] = useState(false);

  function handleAnalyze() {
    if (analyzeEnabled) {
      refetch();
    } else {
      setAnalyzeEnabled(true);
    }
  }

  function handleSave() {
    if (!saveName) return;
    saveModel.mutate({
      name: saveName,
      path: savePath || undefined,
      shard: saveShard || undefined,
      overwrite: saveOverwrite,
    });
  }

  return (
    <div className="space-y-3">
      <Button size="sm" variant="secondary" onClick={handleAnalyze} disabled={isLoading} className="w-full">
        {isLoading && <Loader2 className="h-3.5 w-3.5 animate-spin mr-1.5" />}
        Analyze model
      </Button>

      {analysis && "name" in analysis && (
        <>
          <div className="text-xs space-y-1">
            <p><span className="text-muted-foreground">Name:</span> {analysis.name}</p>
            <p><span className="text-muted-foreground">Type:</span> {analysis.type}</p>
            <p><span className="text-muted-foreground">Class:</span> {analysis.class}</p>
            {analysis.hash && <p><span className="text-muted-foreground">Hash:</span> {analysis.hash}</p>}
          </div>

          {analysis.modules.length > 0 && (
            <div className="border border-border rounded-md overflow-hidden">
              <table className="w-full text-[11px]">
                <thead>
                  <tr className="bg-muted/50 border-b border-border">
                    <th className="px-2 py-1 text-left font-medium">Name</th>
                    <th className="px-2 py-1 text-left font-medium">Class</th>
                    <th className="px-2 py-1 text-left font-medium">Device</th>
                    <th className="px-2 py-1 text-left font-medium">Dtype</th>
                    <th className="px-2 py-1 text-right font-medium">Params</th>
                  </tr>
                </thead>
                <tbody>
                  {analysis.modules.map((m) => (
                    <tr key={m.name} className="border-b border-border/50 hover:bg-muted/30">
                      <td className="px-2 py-1 font-mono truncate max-w-[100px]">{m.name}</td>
                      <td className="px-2 py-1 truncate max-w-[100px]">{m.cls}</td>
                      <td className="px-2 py-1">{m.device ?? "-"}</td>
                      <td className="px-2 py-1">{m.quant ?? m.dtype ?? "-"}</td>
                      <td className="px-2 py-1 text-right font-mono">{m.params ? formatParams(m.params) : "-"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* Save section */}
          <button type="button" onClick={() => setShowSave(!showSave)} className="flex items-center gap-1 text-xs font-medium text-muted-foreground hover:text-foreground w-full">
            {showSave ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
            Save model
          </button>
          {showSave && (
            <div className="space-y-2 pl-4">
              <div>
                <Label className="text-[11px]">Name</Label>
                <Input className="h-7 text-xs" value={saveName} onChange={(e) => setSaveName(e.target.value)} placeholder="model-name" />
              </div>
              <div>
                <Label className="text-[11px]">Path</Label>
                <Input className="h-7 text-xs" value={savePath} onChange={(e) => setSavePath(e.target.value)} placeholder="diffusers directory" />
              </div>
              <div>
                <Label className="text-[11px]">Shard size</Label>
                <Input className="h-7 text-xs" value={saveShard} onChange={(e) => setSaveShard(e.target.value)} placeholder="10GB" />
              </div>
              <div className="flex items-center gap-2">
                <Checkbox id="save-overwrite" checked={saveOverwrite} onCheckedChange={(v) => setSaveOverwrite(!!v)} />
                <Label htmlFor="save-overwrite" className="text-[11px]">Overwrite</Label>
              </div>
              <Button size="sm" variant="secondary" onClick={handleSave} disabled={saveModel.isPending || !saveName} className="w-full">
                {saveModel.isPending && <Loader2 className="h-3 w-3 animate-spin mr-1" />}
                Save
              </Button>
              {saveModel.data && <p className="text-[11px] text-muted-foreground">{saveModel.data.status}</p>}
            </div>
          )}

          {/* Metadata section */}
          {analysis.meta && Object.keys(analysis.meta).length > 0 && (
            <>
              <button type="button" onClick={() => setShowMeta(!showMeta)} className="flex items-center gap-1 text-xs font-medium text-muted-foreground hover:text-foreground w-full">
                {showMeta ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
                Metadata ({Object.keys(analysis.meta).length} keys)
              </button>
              {showMeta && (
                <pre className="text-[10px] bg-muted/30 rounded-md p-2 overflow-auto max-h-[300px] whitespace-pre-wrap break-all">
                  {JSON.stringify(analysis.meta, null, 2)}
                </pre>
              )}
            </>
          )}
        </>
      )}

      {analyzeEnabled && !isLoading && (!analysis || !("name" in analysis)) && (
        <p className="text-xs text-muted-foreground">No model loaded or analysis returned empty.</p>
      )}
    </div>
  );
}
