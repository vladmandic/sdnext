import { useState } from "react";
import { Loader2, ChevronDown, ChevronRight } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/client";
import { useCivitaiDownload } from "@/api/hooks/useModelOps";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Combobox } from "@/components/ui/combobox";

interface CivitSearchResult {
  id: number;
  name: string;
  tags: string;
  downloads: number;
  rating: number;
}

const MODEL_TYPES = ["Model", "LoRA", "Embedding", "VAE", "Other"] as const;

export function CivitaiSubTab() {
  const [keyword, setKeyword] = useState("");
  const [tag, setTag] = useState("");
  const [modelType, setModelType] = useState("Model");
  const [searchEnabled, setSearchEnabled] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [downloadUrl, setDownloadUrl] = useState("");
  const [downloadName, setDownloadName] = useState("");
  const [downloadPath, setDownloadPath] = useState("");
  const [downloadToken, setDownloadToken] = useState("");
  const civitDownload = useCivitaiDownload();

  const { data: results, isLoading, refetch } = useQuery({
    queryKey: ["civitai-search", keyword, tag, modelType],
    queryFn: () =>
      api.get<{ status: string; data: CivitSearchResult[] }>("/sdapi/v1/civitai", {
        name: keyword,
        tag,
        model_type: modelType,
      }),
    enabled: searchEnabled && keyword.length > 0,
    staleTime: 30_000,
  });

  function handleSearch() {
    if (!keyword) return;
    if (searchEnabled) {
      refetch();
    } else {
      setSearchEnabled(true);
    }
  }

  function handleDownload() {
    if (!downloadUrl) return;
    civitDownload.mutate({
      url: downloadUrl,
      name: downloadName || undefined,
      path: downloadPath || undefined,
      model_type: modelType,
      token: downloadToken || undefined,
    });
  }

  const searchResults = results?.data ?? [];

  return (
    <div className="space-y-3">
      <div className="flex gap-2">
        <Input
          placeholder="Search CivitAI..."
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

      <div className="flex gap-2 items-end">
        <div className="flex-1">
          <Label className="text-[11px]">Type</Label>
          <Combobox value={modelType} onValueChange={setModelType} options={MODEL_TYPES} className="h-7 text-xs" />
        </div>
        <div className="flex-1">
          <Label className="text-[11px]">Tag</Label>
          <Input className="h-7 text-xs" value={tag} onChange={(e) => setTag(e.target.value)} placeholder="optional tag" />
        </div>
      </div>

      {searchResults.length > 0 && (
        <div className="border border-border rounded-md overflow-auto max-h-[250px]">
          <table className="w-full text-[11px]">
            <thead>
              <tr className="bg-muted/50 border-b border-border">
                <th className="px-2 py-1 text-left font-medium">Name</th>
                <th className="px-2 py-1 text-left font-medium">Tags</th>
                <th className="px-2 py-1 text-right font-medium">Downloads</th>
                <th className="px-2 py-1 text-right font-medium">Rating</th>
              </tr>
            </thead>
            <tbody>
              {searchResults.map((r) => (
                <tr key={r.id} className="border-b border-border/50 hover:bg-muted/30">
                  <td className="px-2 py-1 truncate max-w-[140px]">{r.name}</td>
                  <td className="px-2 py-1 truncate max-w-[100px]">{r.tags}</td>
                  <td className="px-2 py-1 text-right font-mono">{r.downloads}</td>
                  <td className="px-2 py-1 text-right font-mono">{r.rating?.toFixed(1) ?? "-"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="border border-border rounded-md p-2 space-y-2">
        <p className="text-xs font-medium">Download from CivitAI</p>
        <div>
          <Label className="text-[11px]">Model URL</Label>
          <Input className="h-7 text-xs" value={downloadUrl} onChange={(e) => setDownloadUrl(e.target.value)} placeholder="https://civitai.com/api/download/..." />
        </div>

        <button type="button" onClick={() => setShowAdvanced(!showAdvanced)} className="flex items-center gap-1 text-[11px] text-muted-foreground hover:text-foreground">
          {showAdvanced ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
          Advanced
        </button>
        {showAdvanced && (
          <div className="space-y-2 pl-3">
            <div>
              <Label className="text-[11px]">Filename</Label>
              <Input className="h-7 text-xs" value={downloadName} onChange={(e) => setDownloadName(e.target.value)} placeholder="auto-detect" />
            </div>
            <div>
              <Label className="text-[11px]">Path</Label>
              <Input className="h-7 text-xs" value={downloadPath} onChange={(e) => setDownloadPath(e.target.value)} placeholder="auto by type" />
            </div>
            <div>
              <Label className="text-[11px]">Token</Label>
              <Input className="h-7 text-xs" type="password" value={downloadToken} onChange={(e) => setDownloadToken(e.target.value)} placeholder="CivitAI token" />
            </div>
          </div>
        )}

        <Button size="sm" variant="secondary" onClick={handleDownload} disabled={civitDownload.isPending || !downloadUrl} className="w-full">
          {civitDownload.isPending && <Loader2 className="h-3 w-3 animate-spin mr-1" />}
          Download
        </Button>
        {civitDownload.data && <p className="text-[11px] text-muted-foreground">{civitDownload.data.status}</p>}
      </div>
    </div>
  );
}
