import { useState } from "react";
import { Loader2, ExternalLink } from "lucide-react";
import { useLoaderPipelines, useLoaderComponents, useLoaderLoad } from "@/api/hooks/useModelOps";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Combobox } from "@/components/ui/combobox";
import type { LoaderComponentsResponse } from "@/api/types/modelOps";

export function LoaderSubTab() {
  const { data: pipelineData } = useLoaderPipelines();
  const [modelType, setModelType] = useState("Autodetect");
  const fetchEnabled = modelType !== "" && modelType !== "Autodetect";
  const { data: compData, isLoading: compLoading } = useLoaderComponents(modelType, fetchEnabled);

  const pipelines = pipelineData?.pipelines ?? [];

  return (
    <div className="space-y-3">
      <p className="text-[11px] text-muted-foreground">Custom model loader — select pipeline and configure components.</p>

      <div>
        <Label className="text-[11px]">Pipeline type</Label>
        <Combobox value={modelType} onValueChange={setModelType} options={pipelines} className="h-7 text-xs" />
      </div>

      {compLoading && <p className="text-xs text-muted-foreground">Loading components...</p>}

      {compData && <LoaderEditor key={modelType} compData={compData} modelType={modelType} />}
    </div>
  );
}

function LoaderEditor({ compData, modelType }: { compData: LoaderComponentsResponse; modelType: string }) {
  const load = useLoaderLoad();
  const [repo, setRepo] = useState(compData.repo ?? "");
  const [editedComponents, setEditedComponents] = useState<Record<number, { local?: string; remote?: string; dtype?: string; quant?: boolean }>>({});

  const loadableComponents = (compData.components ?? []).filter((c) => c.loadable);

  function getEditedValue(id: number, field: "local" | "remote" | "dtype" | "quant") {
    return editedComponents[id]?.[field];
  }

  function setComponentField(id: number, field: "local" | "remote" | "dtype" | "quant", value: string | boolean) {
    setEditedComponents((prev) => ({
      ...prev,
      [id]: { ...prev[id], [field]: value },
    }));
  }

  function handleLoad() {
    if (!repo) return;
    const components = loadableComponents.map((c) => ({
      id: c.id,
      local: getEditedValue(c.id, "local") as string | undefined ?? c.local ?? undefined,
      remote: getEditedValue(c.id, "remote") as string | undefined ?? c.remote ?? undefined,
      dtype: getEditedValue(c.id, "dtype") as string | undefined ?? c.dtype ?? undefined,
      quant: getEditedValue(c.id, "quant") as boolean | undefined ?? c.quant ?? undefined,
    }));
    load.mutate({ model_type: modelType, repo, components });
  }

  return (
    <>
      <div className="text-xs space-y-1">
        <p><span className="text-muted-foreground">Class:</span> {compData.class}</p>
      </div>

      <div>
        <Label className="text-[11px]">Repo</Label>
        <div className="flex gap-1 items-center">
          <Input className="h-7 text-xs flex-1" value={repo} onChange={(e) => setRepo(e.target.value)} placeholder="HuggingFace repo ID" />
          {repo && (
            <a href={`https://huggingface.co/${repo}`} target="_blank" rel="noopener noreferrer" className="text-muted-foreground hover:text-foreground shrink-0">
              <ExternalLink className="h-3.5 w-3.5" />
            </a>
          )}
        </div>
      </div>

      {loadableComponents.length > 0 && (
        <div className="border border-border rounded-md overflow-auto">
          <table className="w-full text-[11px]">
            <thead>
              <tr className="bg-muted/50 border-b border-border">
                <th className="px-2 py-1 text-left font-medium">Name</th>
                <th className="px-2 py-1 text-left font-medium">Class</th>
                <th className="px-2 py-1 text-left font-medium">Local</th>
                <th className="px-2 py-1 text-left font-medium">Dtype</th>
              </tr>
            </thead>
            <tbody>
              {loadableComponents.map((c) => (
                <tr key={c.id} className="border-b border-border/50">
                  <td className="px-2 py-1 font-mono truncate max-w-[80px]">{c.name}</td>
                  <td className="px-2 py-1 truncate max-w-[100px]">{c.class_name}</td>
                  <td className="px-1 py-0.5">
                    <Input
                      className="h-6 text-[10px] border-0 bg-transparent p-1"
                      value={(getEditedValue(c.id, "local") as string | undefined) ?? c.local ?? ""}
                      onChange={(e) => setComponentField(c.id, "local", e.target.value)}
                      placeholder="local path"
                    />
                  </td>
                  <td className="px-1 py-0.5">
                    <Input
                      className="h-6 text-[10px] border-0 bg-transparent p-1 w-16"
                      value={(getEditedValue(c.id, "dtype") as string | undefined) ?? c.dtype ?? ""}
                      onChange={(e) => setComponentField(c.id, "dtype", e.target.value)}
                      placeholder="dtype"
                    />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="flex gap-2">
        <Button size="sm" variant="default" onClick={handleLoad} disabled={load.isPending || !repo} className="flex-1">
          {load.isPending && <Loader2 className="h-3 w-3 animate-spin mr-1" />}
          Load model
        </Button>
      </div>
      {load.data && <p className="text-[11px] text-muted-foreground">{load.data.status}</p>}
    </>
  );
}
