import { useState } from "react";
import { Loader2, RefreshCw } from "lucide-react";
import { useLoraLoaded, useLoraExtract } from "@/api/hooks/useModelOps";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { Slider } from "@/components/ui/slider";

export function ExtractLoraSubTab() {
  const { data: loraData, refetch: refetchLoras } = useLoraLoaded();
  const extract = useLoraExtract();
  const [filename, setFilename] = useState("");
  const [maxRank, setMaxRank] = useState(64);
  const [autoRank, setAutoRank] = useState(false);
  const [rankRatio, setRankRatio] = useState(0.5);
  const [includeTe, setIncludeTe] = useState(true);
  const [includeUnet, setIncludeUnet] = useState(true);
  const [overwrite, setOverwrite] = useState(false);

  const loras = loraData?.loras ?? [];

  function handleExtract() {
    if (!filename) return;
    const modules: string[] = [];
    if (includeTe) modules.push("te");
    if (includeUnet) modules.push("unet");
    extract.mutate({
      filename,
      max_rank: maxRank,
      auto_rank: autoRank,
      rank_ratio: autoRank ? rankRatio : undefined,
      modules,
      overwrite,
    });
  }

  return (
    <div className="space-y-3">
      <p className="text-2xs text-muted-foreground">Extract LoRA from loaded model differences.</p>

      <div className="space-y-1">
        <div className="flex items-center justify-between">
          <p className="text-xs font-medium">Loaded LoRAs</p>
          <button type="button" onClick={() => refetchLoras()} className="p-1 text-muted-foreground hover:text-foreground">
            <RefreshCw className="h-3 w-3" />
          </button>
        </div>
        {loras.length > 0 ? (
          <div className="space-y-0.5">
            {loras.map((name) => (
              <p key={name} className="text-2xs font-mono px-2 py-0.5 bg-muted/30 rounded">{name}</p>
            ))}
          </div>
        ) : (
          <p className="text-2xs text-muted-foreground">No LoRAs currently loaded in model.</p>
        )}
      </div>

      <div>
        <Label className="text-2xs">Output filename</Label>
        <Input className="h-6 text-2xs" value={filename} onChange={(e) => setFilename(e.target.value)} placeholder="extracted-lora" />
      </div>

      <div className="space-y-1">
        <p className="text-xs font-medium">Modules to extract</p>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <Checkbox id="extract-te" checked={includeTe} onCheckedChange={(v) => setIncludeTe(!!v)} />
            <Label htmlFor="extract-te" className="text-2xs">Text Encoder</Label>
          </div>
          <div className="flex items-center gap-2">
            <Checkbox id="extract-unet" checked={includeUnet} onCheckedChange={(v) => setIncludeUnet(!!v)} />
            <Label htmlFor="extract-unet" className="text-2xs">UNet</Label>
          </div>
        </div>
      </div>

      <div>
        <Label className="text-2xs">Max rank: {maxRank}</Label>
        <Slider value={[maxRank]} onValueChange={([v]) => setMaxRank(v)} min={1} max={256} step={1} className="mt-1" />
      </div>

      <div className="space-y-1">
        <div className="flex items-center gap-2">
          <Checkbox id="extract-autorank" checked={autoRank} onCheckedChange={(v) => setAutoRank(!!v)} />
          <Label htmlFor="extract-autorank" className="text-2xs">Auto rank</Label>
        </div>
        {autoRank && (
          <div className="pl-5">
            <Label className="text-2xs">Rank ratio: {rankRatio.toFixed(2)}</Label>
            <Slider value={[rankRatio]} onValueChange={([v]) => setRankRatio(v)} min={0} max={1} step={0.05} className="mt-1" />
          </div>
        )}
      </div>

      <div className="flex items-center gap-2">
        <Checkbox id="extract-overwrite" checked={overwrite} onCheckedChange={(v) => setOverwrite(!!v)} />
        <Label htmlFor="extract-overwrite" className="text-2xs">Overwrite existing</Label>
      </div>

      <Button size="sm" variant="default" onClick={handleExtract} disabled={extract.isPending || !filename} className="w-full">
        {extract.isPending && <Loader2 className="h-3 w-3 animate-spin mr-1" />}
        Extract LoRA
      </Button>
      {extract.data && <p className="text-2xs text-muted-foreground">{extract.data.status}</p>}
    </div>
  );
}
