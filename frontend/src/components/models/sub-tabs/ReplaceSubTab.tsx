import { useState } from "react";
import { Loader2 } from "lucide-react";
import { useReplaceComponents } from "@/api/hooks/useModelOps";
import { useModelList, useSamplerList } from "@/api/hooks/useModels";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { Combobox } from "@/components/ui/combobox";

const MODEL_TYPES = ["sdxl"] as const;
const PRECISION_OPTIONS = ["fp32", "fp16", "bf16"] as const;
const PREDICTION_OPTIONS = ["", "epsilon", "v"] as const;

export function ReplaceSubTab() {
  const { data: models } = useModelList();
  const { data: samplers } = useSamplerList();
  const replace = useReplaceComponents();

  const [modelType, setModelType] = useState("sdxl");
  const [modelName, setModelName] = useState("");
  const [customName, setCustomName] = useState("");
  const [compUnet, setCompUnet] = useState("");
  const [compVae, setCompVae] = useState("");
  const [compTe1, setCompTe1] = useState("");
  const [compTe2, setCompTe2] = useState("");
  const [precision, setPrecision] = useState("fp16");
  const [scheduler, setScheduler] = useState("");
  const [prediction, setPrediction] = useState("");
  const [compLora, setCompLora] = useState("");
  const [compFuse, setCompFuse] = useState("1.0");
  const [metaAuthor, setMetaAuthor] = useState("");
  const [metaVersion, setMetaVersion] = useState("");
  const [metaLicense, setMetaLicense] = useState("");
  const [metaDesc, setMetaDesc] = useState("");
  const [metaHint, setMetaHint] = useState("");
  const [createDiffusers, setCreateDiffusers] = useState(true);
  const [createSafetensors, setCreateSafetensors] = useState(false);

  const modelOptions = (models ?? []).map((m) => ({ value: m.title, label: m.model_name }));
  const samplerOptions = [{ value: "", label: "(None)" }, ...(samplers ?? []).map((s) => ({ value: s.name, label: s.name }))];

  function handleReplace() {
    if (!customName || !modelName) return;
    replace.mutate({
      model_type: modelType,
      model_name: modelName,
      custom_name: customName,
      comp_unet: compUnet || undefined,
      comp_vae: compVae || undefined,
      comp_te1: compTe1 || undefined,
      comp_te2: compTe2 || undefined,
      precision,
      comp_scheduler: scheduler || undefined,
      comp_prediction: prediction || undefined,
      comp_lora: compLora || undefined,
      comp_fuse: parseFloat(compFuse) || undefined,
      meta_author: metaAuthor || undefined,
      meta_version: metaVersion || undefined,
      meta_license: metaLicense || undefined,
      meta_desc: metaDesc || undefined,
      meta_hint: metaHint || undefined,
      create_diffusers: createDiffusers,
      create_safetensors: createSafetensors,
    });
  }

  return (
    <div className="space-y-3">
      <p className="text-2xs text-muted-foreground">Replace model components (SDXL only).</p>

      <div className="flex gap-2">
        <div className="flex-1">
          <Label className="text-2xs">Model type</Label>
          <Combobox value={modelType} onValueChange={setModelType} options={MODEL_TYPES} className="h-6 text-2xs" />
        </div>
        <div className="flex-1">
          <Label className="text-2xs">Precision</Label>
          <Combobox value={precision} onValueChange={setPrecision} options={PRECISION_OPTIONS} className="h-6 text-2xs" />
        </div>
      </div>

      <div>
        <Label className="text-2xs">Input model</Label>
        <Combobox value={modelName} onValueChange={setModelName} options={modelOptions} placeholder="Select model..." className="h-6 text-2xs" />
      </div>

      <div>
        <Label className="text-2xs">Output name</Label>
        <Input className="h-6 text-2xs" value={customName} onChange={(e) => setCustomName(e.target.value)} placeholder="output-model" />
      </div>

      <div className="space-y-2">
        <p className="text-xs font-medium">Components</p>
        <div>
          <Label className="text-2xs">UNet</Label>
          <Input className="h-6 text-2xs" value={compUnet} onChange={(e) => setCompUnet(e.target.value)} placeholder="path or repo" />
        </div>
        <div>
          <Label className="text-2xs">VAE</Label>
          <Input className="h-6 text-2xs" value={compVae} onChange={(e) => setCompVae(e.target.value)} placeholder="path or repo" />
        </div>
        <div>
          <Label className="text-2xs">Text Encoder 1</Label>
          <Input className="h-6 text-2xs" value={compTe1} onChange={(e) => setCompTe1(e.target.value)} placeholder="path or repo" />
        </div>
        <div>
          <Label className="text-2xs">Text Encoder 2</Label>
          <Input className="h-6 text-2xs" value={compTe2} onChange={(e) => setCompTe2(e.target.value)} placeholder="path or repo" />
        </div>
      </div>

      <div className="space-y-2">
        <p className="text-xs font-medium">Settings</p>
        <div>
          <Label className="text-2xs">Scheduler</Label>
          <Combobox value={scheduler} onValueChange={setScheduler} options={samplerOptions} placeholder="(None)" className="h-6 text-2xs" />
        </div>
        <div>
          <Label className="text-2xs">Prediction type</Label>
          <Combobox value={prediction} onValueChange={setPrediction} options={PREDICTION_OPTIONS} placeholder="(None)" className="h-6 text-2xs" />
        </div>
      </div>

      <div className="space-y-2">
        <p className="text-xs font-medium">LoRA merge</p>
        <div>
          <Label className="text-2xs">LoRA list (comma-separated, name:strength)</Label>
          <Input className="h-6 text-2xs" value={compLora} onChange={(e) => setCompLora(e.target.value)} placeholder="lora1:0.8, lora2" />
        </div>
        <div>
          <Label className="text-2xs">Fuse strength</Label>
          <Input className="h-6 text-2xs" value={compFuse} onChange={(e) => setCompFuse(e.target.value)} placeholder="1.0" />
        </div>
      </div>

      <div className="space-y-2">
        <p className="text-xs font-medium">Metadata</p>
        <div className="grid grid-cols-2 gap-2">
          <div>
            <Label className="text-2xs">Author</Label>
            <Input className="h-6 text-2xs" value={metaAuthor} onChange={(e) => setMetaAuthor(e.target.value)} />
          </div>
          <div>
            <Label className="text-2xs">Version</Label>
            <Input className="h-6 text-2xs" value={metaVersion} onChange={(e) => setMetaVersion(e.target.value)} />
          </div>
          <div>
            <Label className="text-2xs">License</Label>
            <Input className="h-6 text-2xs" value={metaLicense} onChange={(e) => setMetaLicense(e.target.value)} />
          </div>
          <div>
            <Label className="text-2xs">Hint</Label>
            <Input className="h-6 text-2xs" value={metaHint} onChange={(e) => setMetaHint(e.target.value)} />
          </div>
        </div>
        <div>
          <Label className="text-2xs">Description</Label>
          <Input className="h-6 text-2xs" value={metaDesc} onChange={(e) => setMetaDesc(e.target.value)} />
        </div>
      </div>

      <div className="space-y-1">
        <p className="text-xs font-medium">Save options</p>
        <div className="flex items-center gap-2">
          <Checkbox id="replace-diffusers" checked={createDiffusers} onCheckedChange={(v) => setCreateDiffusers(!!v)} />
          <Label htmlFor="replace-diffusers" className="text-2xs">Diffusers format</Label>
        </div>
        <div className="flex items-center gap-2">
          <Checkbox id="replace-safetensors" checked={createSafetensors} onCheckedChange={(v) => setCreateSafetensors(!!v)} />
          <Label htmlFor="replace-safetensors" className="text-2xs">Safetensors format</Label>
        </div>
      </div>

      <Button size="sm" variant="default" onClick={handleReplace} disabled={replace.isPending || !customName || !modelName} className="w-full">
        {replace.isPending && <Loader2 className="h-3 w-3 animate-spin mr-1" />}
        Replace & Build
      </Button>
      {replace.data && <p className="text-2xs text-muted-foreground">{replace.data.status}</p>}
    </div>
  );
}
