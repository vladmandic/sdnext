import { useMemo } from "react";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Combobox } from "@/components/ui/combobox";
import { Input } from "@/components/ui/input";
import { NumberInput } from "@/components/ui/number-input";
import { Textarea } from "@/components/ui/textarea";
import { ParamSlider } from "@/components/generation/ParamSlider";
import { usePromptEnhanceModels } from "@/api/hooks/usePromptEnhance";
import { usePromptEnhanceStore } from "@/stores/promptEnhanceStore";
import type { PromptEnhanceModel } from "@/api/types/promptEnhance";

function ModelBadges({ model }: { model: PromptEnhanceModel | undefined }) {
  if (!model) return null;
  return (
    <span className="flex gap-1 ml-1">
      {model.vision && <span className="text-[9px] bg-blue-500/20 text-blue-400 px-1 rounded">VL</span>}
      {model.thinking && <span className="text-[9px] bg-amber-500/20 text-amber-400 px-1 rounded">Think</span>}
    </span>
  );
}

function SwitchRow({ label, checked, onCheckedChange }: { label: string; checked: boolean; onCheckedChange: (v: boolean) => void }) {
  return (
    <div className="flex items-center justify-between">
      <Label className="text-[11px] text-muted-foreground">{label}</Label>
      <Switch size="sm" checked={checked} onCheckedChange={onCheckedChange} />
    </div>
  );
}

export function PromptEnhancePanel() {
  const { data: models } = usePromptEnhanceModels();
  const store = usePromptEnhanceStore();

  const selectedModel = useMemo(
    () => models?.find((m) => m.name === store.model),
    [models, store.model],
  );

  const modelOptions = useMemo(
    () => (models ?? []).map((m) => ({
      value: m.name,
      label: m.name.split("/").pop() ?? m.name,
    })),
    [models],
  );

  return (
    <div className="flex flex-col gap-2.5 pt-2 pb-1">
      {/* Model selector */}
      <div className="flex flex-col gap-1">
        <div className="flex items-center">
          <Label className="text-[11px] text-muted-foreground">Model</Label>
          <ModelBadges model={selectedModel} />
        </div>
        <Combobox
          value={store.model}
          onValueChange={store.setModel}
          options={modelOptions}
          placeholder="Select model..."
          searchPlaceholder="Search models..."
          className="h-7 text-xs"
        />
      </div>

      {/* Conditional toggles */}
      {selectedModel?.vision && (
        <SwitchRow label="Use vision" checked={store.useVision} onCheckedChange={store.setUseVision} />
      )}
      {selectedModel?.thinking && (
        <SwitchRow label="Thinking mode" checked={store.thinking} onCheckedChange={store.setThinking} />
      )}
      {selectedModel?.thinking && store.thinking && (
        <SwitchRow label="Keep thinking" checked={store.keepThinking} onCheckedChange={store.setKeepThinking} />
      )}

      <SwitchRow label="NSFW" checked={store.nsfw} onCheckedChange={store.setNsfw} />

      {/* Sampling parameters */}
      <ParamSlider label="Temp" value={store.temperature} onChange={store.setTemperature} min={0.1} max={2.0} step={0.05} />
      <ParamSlider label="Rep. pen." value={store.repetitionPenalty} onChange={store.setRepetitionPenalty} min={1.0} max={2.0} step={0.05} />

      {/* Number inputs row */}
      <div className="grid grid-cols-3 gap-2">
        <div className="flex flex-col gap-0.5">
          <Label className="text-[10px] text-muted-foreground">Max tokens</Label>
          <NumberInput value={store.maxTokens} onChange={store.setMaxTokens} min={64} max={2048} step={64} fallback={512} className="h-6 text-[11px] text-center px-1" />
        </div>
        <div className="flex flex-col gap-0.5">
          <Label className="text-[10px] text-muted-foreground">Top K</Label>
          <NumberInput value={store.topK} onChange={store.setTopK} min={0} max={200} step={1} fallback={0} className="h-6 text-[11px] text-center px-1" />
        </div>
        <div className="flex flex-col gap-0.5">
          <Label className="text-[10px] text-muted-foreground">Top P</Label>
          <NumberInput value={store.topP} onChange={store.setTopP} min={0} max={1} step={0.05} fallback={0} className="h-6 text-[11px] text-center px-1" />
        </div>
      </div>

      {/* Seed */}
      <div className="flex items-center gap-2">
        <Label className="text-[11px] text-muted-foreground w-16 flex-shrink-0">Seed</Label>
        <NumberInput value={store.seed} onChange={store.setSeed} min={-1} max={4294967294} step={1} fallback={-1} className="h-6 text-[11px] text-center px-1 flex-1" />
      </div>

      {/* Prefix / Suffix */}
      <div className="grid grid-cols-2 gap-2">
        <div className="flex flex-col gap-0.5">
          <Label className="text-[10px] text-muted-foreground">Prefix</Label>
          <Input value={store.prefix} onChange={(e) => store.setPrefix(e.target.value)} placeholder="Prefix..." className="h-6 text-[11px] px-2" />
        </div>
        <div className="flex flex-col gap-0.5">
          <Label className="text-[10px] text-muted-foreground">Suffix</Label>
          <Input value={store.suffix} onChange={(e) => store.setSuffix(e.target.value)} placeholder="Suffix..." className="h-6 text-[11px] px-2" />
        </div>
      </div>

      {/* System prompt */}
      <div className="flex flex-col gap-0.5">
        <Label className="text-[10px] text-muted-foreground">System prompt</Label>
        <Textarea
          value={store.systemPrompt}
          onChange={(e) => store.setSystemPrompt(e.target.value)}
          placeholder="Custom system prompt..."
          className="min-h-[40px] max-h-[80px] resize-y text-[11px]"
        />
      </div>
    </div>
  );
}
