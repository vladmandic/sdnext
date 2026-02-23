import { useState } from "react";
import { Loader2, ChevronDown, ChevronRight } from "lucide-react";
import { useMergeMethods, useMergeModels } from "@/api/hooks/useModelOps";
import { useModelList, useVaeList } from "@/api/hooks/useModels";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { Combobox } from "@/components/ui/combobox";
import { Slider } from "@/components/ui/slider";

const PRECISION_OPTIONS = ["fp16", "fp32"] as const;
const DEVICE_OPTIONS = ["cpu", "gpu", "shuffle"] as const;
const FORMAT_OPTIONS = ["safetensors"] as const;

export function MergeSubTab() {
  const { data: methodsInfo, isLoading: methodsLoading } = useMergeMethods();
  const { data: models } = useModelList();
  const { data: vaes } = useVaeList();
  const merge = useMergeModels();

  const [customName, setCustomName] = useState("");
  const [primary, setPrimary] = useState("");
  const [secondary, setSecondary] = useState("");
  const [tertiary, setTertiary] = useState("");
  const [method, setMethod] = useState("weighted_sum");
  const [alpha, setAlpha] = useState(0.5);
  const [beta, setBeta] = useState(0.5);
  const [precision, setPrecision] = useState("fp16");
  const [device, setDevice] = useState("cpu");
  const [format, setFormat] = useState("safetensors");
  const [overwrite, setOverwrite] = useState(false);
  const [saveMeta, setSaveMeta] = useState(true);
  const [weightsClip, setWeightsClip] = useState(false);
  const [prune, setPrune] = useState(false);
  const [reBasin, setReBasin] = useState(false);
  const [reBasinIter, setReBasinIter] = useState(5);
  const [unload, setUnload] = useState(true);
  const [bakeVae, setBakeVae] = useState("");
  const [showAdvanced, setShowAdvanced] = useState(false);

  const isTriple = methodsInfo?.triple_methods.includes(method) ?? false;
  const isBeta = methodsInfo?.beta_methods.includes(method) ?? false;
  const methodDoc = methodsInfo?.docs[method] ?? "";

  const modelOptions = (models ?? []).map((m) => ({ value: m.title, label: m.model_name }));
  const vaeOptions = [{ value: "", label: "(None)" }, ...(vaes ?? []).map((v) => ({ value: v.model_name, label: v.model_name }))];

  function handleMerge() {
    if (!customName || !primary || !secondary) return;
    merge.mutate({
      custom_name: customName,
      primary_model_name: primary,
      secondary_model_name: secondary,
      merge_mode: method,
      tertiary_model_name: isTriple ? tertiary || undefined : undefined,
      alpha,
      beta: isBeta ? beta : undefined,
      precision,
      checkpoint_format: format,
      save_metadata: saveMeta,
      weights_clip: weightsClip,
      prune,
      re_basin: reBasin,
      re_basin_iterations: reBasin ? reBasinIter : undefined,
      device,
      unload,
      overwrite,
      bake_in_vae: bakeVae || undefined,
    });
  }

  if (methodsLoading) return <p className="text-xs text-muted-foreground">Loading merge methods...</p>;

  return (
    <div className="space-y-3">
      <div>
        <Label className="text-2xs">Output name</Label>
        <Input className="h-6 text-2xs" value={customName} onChange={(e) => setCustomName(e.target.value)} placeholder="merged-model" />
      </div>

      <div>
        <Label className="text-2xs">Method</Label>
        <Combobox value={method} onValueChange={setMethod} options={methodsInfo?.methods ?? []} className="h-6 text-2xs" />
        {methodDoc && <p className="text-3xs text-muted-foreground mt-1">{methodDoc.split("\n")[0]}</p>}
      </div>

      <div>
        <Label className="text-2xs">Primary model (A)</Label>
        <Combobox value={primary} onValueChange={setPrimary} options={modelOptions} placeholder="Select model A..." className="h-6 text-2xs" />
      </div>

      <div>
        <Label className="text-2xs">Secondary model (B)</Label>
        <Combobox value={secondary} onValueChange={setSecondary} options={modelOptions} placeholder="Select model B..." className="h-6 text-2xs" />
      </div>

      {isTriple && (
        <div>
          <Label className="text-2xs">Tertiary model (C)</Label>
          <Combobox value={tertiary} onValueChange={setTertiary} options={modelOptions} placeholder="Select model C..." className="h-6 text-2xs" />
        </div>
      )}

      <div>
        <Label className="text-2xs">Alpha: {alpha.toFixed(2)}</Label>
        <Slider value={[alpha]} onValueChange={([v]) => setAlpha(v)} min={0} max={1} step={0.05} className="mt-1" />
      </div>

      {isBeta && (
        <div>
          <Label className="text-2xs">Beta: {beta.toFixed(2)}</Label>
          <Slider value={[beta]} onValueChange={([v]) => setBeta(v)} min={0} max={1} step={0.05} className="mt-1" />
        </div>
      )}

      <button type="button" onClick={() => setShowAdvanced(!showAdvanced)} className="flex items-center gap-1 text-xs font-medium text-muted-foreground hover:text-foreground w-full">
        {showAdvanced ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
        Advanced options
      </button>
      {showAdvanced && (
        <div className="space-y-2 pl-3">
          <div className="flex gap-2">
            <div className="flex-1">
              <Label className="text-2xs">Precision</Label>
              <Combobox value={precision} onValueChange={setPrecision} options={PRECISION_OPTIONS} className="h-6 text-2xs" />
            </div>
            <div className="flex-1">
              <Label className="text-2xs">Device</Label>
              <Combobox value={device} onValueChange={setDevice} options={DEVICE_OPTIONS} className="h-6 text-2xs" />
            </div>
          </div>
          <div>
            <Label className="text-2xs">Format</Label>
            <Combobox value={format} onValueChange={setFormat} options={FORMAT_OPTIONS} className="h-6 text-2xs" />
          </div>
          <div>
            <Label className="text-2xs">Bake-in VAE</Label>
            <Combobox value={bakeVae} onValueChange={setBakeVae} options={vaeOptions} placeholder="(None)" className="h-6 text-2xs" />
          </div>
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <Checkbox id="merge-overwrite" checked={overwrite} onCheckedChange={(v) => setOverwrite(!!v)} />
              <Label htmlFor="merge-overwrite" className="text-2xs">Overwrite</Label>
            </div>
            <div className="flex items-center gap-2">
              <Checkbox id="merge-savemeta" checked={saveMeta} onCheckedChange={(v) => setSaveMeta(!!v)} />
              <Label htmlFor="merge-savemeta" className="text-2xs">Save metadata</Label>
            </div>
            <div className="flex items-center gap-2">
              <Checkbox id="merge-wclip" checked={weightsClip} onCheckedChange={(v) => setWeightsClip(!!v)} />
              <Label htmlFor="merge-wclip" className="text-2xs">Weights clip</Label>
            </div>
            <div className="flex items-center gap-2">
              <Checkbox id="merge-prune" checked={prune} onCheckedChange={(v) => setPrune(!!v)} />
              <Label htmlFor="merge-prune" className="text-2xs">Prune</Label>
            </div>
            <div className="flex items-center gap-2">
              <Checkbox id="merge-unload" checked={unload} onCheckedChange={(v) => setUnload(!!v)} />
              <Label htmlFor="merge-unload" className="text-2xs">Unload current model</Label>
            </div>
            <div className="flex items-center gap-2">
              <Checkbox id="merge-rebasin" checked={reBasin} onCheckedChange={(v) => setReBasin(!!v)} />
              <Label htmlFor="merge-rebasin" className="text-2xs">ReBasin</Label>
            </div>
            {reBasin && (
              <div className="pl-5">
                <Label className="text-2xs">Iterations: {reBasinIter}</Label>
                <Slider value={[reBasinIter]} onValueChange={([v]) => setReBasinIter(v)} min={1} max={20} step={1} className="mt-1" />
              </div>
            )}
          </div>
        </div>
      )}

      <Button size="sm" variant="default" onClick={handleMerge} disabled={merge.isPending || !customName || !primary || !secondary} className="w-full">
        {merge.isPending && <Loader2 className="h-3 w-3 animate-spin mr-1" />}
        Merge
      </Button>
      {merge.data && <p className="text-2xs text-muted-foreground">{merge.data.status}</p>}
    </div>
  );
}
