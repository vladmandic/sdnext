import { useCallback, useMemo } from "react";
import { Lightbulb, Eye } from "lucide-react";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Combobox, type ComboboxGroup, type ComboboxOption } from "@/components/ui/combobox";
import { Input } from "@/components/ui/input";
import { NumberInput } from "@/components/ui/number-input";
import { Textarea } from "@/components/ui/textarea";
import { ParamSlider } from "@/components/generation/ParamSlider";
import { ParamGrid } from "@/components/generation/ParamRow";
import { usePromptEnhanceModels } from "@/api/hooks/usePromptEnhance";
import { usePromptEnhanceStore } from "@/stores/promptEnhanceStore";
import { stripPua } from "@/lib/utils";
import type { PromptEnhanceModel } from "@/api/types/promptEnhance";

function SwitchRow({ label, checked, onCheckedChange }: { label: string; checked: boolean; onCheckedChange: (v: boolean) => void }) {
  return (
    <div className="flex items-center justify-between">
      <Label className="text-2xs text-muted-foreground">{label}</Label>
      <Switch size="sm" checked={checked} onCheckedChange={onCheckedChange} />
    </div>
  );
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return <Label className="text-3xs uppercase tracking-wider text-muted-foreground/60 pt-1">{children}</Label>;
}

export function PromptEnhancePanel() {
  const { data: models } = usePromptEnhanceModels();
  const store = usePromptEnhanceStore();

  const selectedModel = useMemo(
    () => models?.find((m) => m.name === store.model),
    [models, store.model],
  );

  const enhanceGroups = useMemo<ComboboxGroup[]>(() => {
    if (!models?.length) return [];
    const order = ["Gemma", "Gemma Finetunes", "Qwen3.5", "Qwen3.5 Finetunes", "Qwen3", "Qwen3-VL", "Qwen3-VL Finetunes", "Qwen2.5", "Qwen2.5-VL", "Qwen2.5-VL Finetunes", "Qwen2-VL", "Qwen2", "Qwen", "Llama", "SmolLM", "Phi", "Mistral", "Mistral Finetunes", "Other"];
    const buckets: Record<string, ComboboxOption[]> = {};
    for (const m of models) {
      const opt: ComboboxOption = { value: m.name, label: m.name.split("/").pop() ?? m.name };
      (buckets[m.group] ??= []).push(opt);
    }
    for (const opts of Object.values(buckets)) opts.sort((a, b) => {
      const la = typeof a === "string" ? a : a.label;
      const lb = typeof b === "string" ? b : b.label;
      return la.localeCompare(lb, undefined, { numeric: true });
    });
    return order.filter((g) => buckets[g]?.length).map((g) => ({ heading: g, options: buckets[g] }));
  }, [models]);

  const modelsByName = useMemo(() => {
    const map = new Map<string, PromptEnhanceModel>();
    for (const m of models ?? []) map.set(m.name, m);
    return map;
  }, [models]);

  const renderModelLabel = useCallback((value: string, label: string) => {
    const model = modelsByName.get(value);
    return (
      <span className="inline-flex items-center gap-0.5">
        {stripPua(label)}
        {model?.thinking && <Lightbulb className="shrink-0 size-[1em]" />}
        {model?.vision && <Eye className="shrink-0 size-[1em]" />}
      </span>
    );
  }, [modelsByName]);

  return (
    <div className="flex flex-col gap-2.5">
      {/* ── Model ── */}
      <SectionLabel>Model</SectionLabel>
      <div className="flex flex-col gap-1">
        <Label className="text-2xs text-muted-foreground">Model</Label>
        <Combobox
          value={store.model}
          onValueChange={store.setModel}
          groups={enhanceGroups}
          placeholder="Select model..."
          searchPlaceholder="Search models..."
          className="h-6 text-2xs"
          renderLabel={renderModelLabel}
        />
      </div>

      {/* ── Prompts ── */}
      <SectionLabel>Prompts</SectionLabel>
      <div className="flex flex-col gap-0.5">
        <Label className="text-3xs text-muted-foreground">System prompt</Label>
        <Textarea
          value={store.systemPrompt}
          onChange={(e) => store.setSystemPrompt(e.target.value)}
          placeholder="Custom system prompt..."
          className="min-h-10 max-h-20 resize-y text-2xs"
        />
      </div>
      <div className="grid grid-cols-2 gap-2">
        <div className="flex flex-col gap-0.5">
          <Label className="text-3xs text-muted-foreground">Prefix</Label>
          <Input value={store.prefix} onChange={(e) => store.setPrefix(e.target.value)} placeholder="Prefix..." className="h-6 text-2xs px-2" />
        </div>
        <div className="flex flex-col gap-0.5">
          <Label className="text-3xs text-muted-foreground">Suffix</Label>
          <Input value={store.suffix} onChange={(e) => store.setSuffix(e.target.value)} placeholder="Suffix..." className="h-6 text-2xs px-2" />
        </div>
      </div>

      {/* ── Options ── */}
      <SectionLabel>Options</SectionLabel>
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
      <div className="flex flex-col gap-0.5">
        <Label className="text-3xs text-muted-foreground">Prefill</Label>
        <Input value={store.prefill} onChange={(e) => store.setPrefill(e.target.value)} placeholder="Prefill model response..." className="h-6 text-2xs px-2" />
      </div>
      {store.prefill.length > 0 && (
        <SwitchRow label="Keep prefill" checked={store.keepPrefill} onCheckedChange={store.setKeepPrefill} />
      )}

      {/* ── Sampling ── */}
      <SectionLabel>Sampling</SectionLabel>
      <SwitchRow label="Do sample" checked={store.doSample} onCheckedChange={store.setDoSample} />
      <div className={store.doSample ? "" : "opacity-40 pointer-events-none"}>
        <div className="flex flex-col gap-2.5">
          <ParamGrid>
            <ParamSlider label="Temp" value={store.temperature} onChange={store.setTemperature} min={0.1} max={2.0} step={0.05} />
            <ParamSlider label="Rep. pen." value={store.repetitionPenalty} onChange={store.setRepetitionPenalty} min={1.0} max={2.0} step={0.05} />
          </ParamGrid>
          <div className="grid grid-cols-3 gap-2">
            <div className="flex flex-col gap-0.5">
              <Label className="text-3xs text-muted-foreground">Max tokens</Label>
              <NumberInput value={store.maxTokens} onChange={store.setMaxTokens} min={64} max={2048} step={64} fallback={512} className="h-6 text-2xs text-center px-1" />
            </div>
            <div className="flex flex-col gap-0.5">
              <Label className="text-3xs text-muted-foreground">Top K</Label>
              <NumberInput value={store.topK} onChange={store.setTopK} min={0} max={200} step={1} fallback={0} className="h-6 text-2xs text-center px-1" />
            </div>
            <div className="flex flex-col gap-0.5">
              <Label className="text-3xs text-muted-foreground">Top P</Label>
              <NumberInput value={store.topP} onChange={store.setTopP} min={0} max={1} step={0.05} fallback={0} className="h-6 text-2xs text-center px-1" />
            </div>
          </div>
        </div>
      </div>

      {/* Seed (always active) */}
      <div className="flex items-center gap-2">
        <Label className="text-2xs text-muted-foreground w-16 flex-shrink-0">Seed</Label>
        <NumberInput value={store.seed} onChange={store.setSeed} min={-1} max={4294967294} step={1} fallback={-1} className="h-6 text-2xs text-center px-1 flex-1" />
      </div>
    </div>
  );
}
