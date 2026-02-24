import { useCallback, useMemo } from "react";
import { Lightbulb } from "lucide-react";
import { Label } from "@/components/ui/label";
import { Combobox } from "@/components/ui/combobox";
import { Textarea } from "@/components/ui/textarea";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { ParamSection } from "@/components/generation/ParamSection";
import { useCaptionSettingsStore } from "@/stores/captionSettingsStore";
import { useVlmModels } from "@/api/hooks/useCaption";
import { VLM_DEFAULT, CUSTOM_PROMPT_TASKS } from "@/lib/captionModels";
import { stripPua } from "@/lib/utils";
import type { VlmModel } from "@/api/types/caption";

export function VlmSettings() {
  const s = useCaptionSettingsStore((st) => st.vlm);
  const set = useCaptionSettingsStore((st) => st.setVlm);

  const { data: models } = useVlmModels();
  const modelNames = useMemo(() => models?.map((m) => m.name) ?? [], [models]);
  const selectedModel = useMemo(() => models?.find((m) => m.name === s.model), [models, s.model]);
  const prompts = useMemo(() => selectedModel?.prompts ?? ["Use Prompt", "Short Caption", "Normal Caption", "Long Caption"], [selectedModel]);

  const capsByName = useMemo(() => {
    const map = new Map<string, VlmModel>();
    for (const m of models ?? []) map.set(m.name, m);
    return map;
  }, [models]);

  const renderModelLabel = useCallback((value: string, label: string) => {
    const model = capsByName.get(value);
    const caps = model?.capabilities ?? [];
    return (
      <span className="inline-flex items-center gap-0.5">
        {stripPua(label)}
        {caps.includes("thinking") && <Lightbulb className="shrink-0 size-[1em]" />}
      </span>
    );
  }, [capsByName]);

  const handleModelChange = (v: string) => {
    set({ model: v });
    const newModel = models?.find((m) => m.name === v);
    const newPrompts = newModel?.prompts ?? prompts;
    if (!newPrompts.includes(s.task)) {
      set({ task: "Normal Caption" });
    }
  };

  const handleTaskChange = (v: string) => set({ task: v });

  const showCustomPrompt = CUSTOM_PROMPT_TASKS.includes(s.task);

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-col gap-1.5">
        <Label className="text-xs">Model</Label>
        <Combobox
          value={s.model}
          onValueChange={handleModelChange}
          options={modelNames.length > 0 ? modelNames : [VLM_DEFAULT]}
          placeholder={modelNames.length === 0 ? "Loading..." : "Select model"}
          className="w-full text-xs"
          renderLabel={renderModelLabel}
        />
      </div>

      <div className="flex flex-col gap-1.5">
        <Label className="text-xs">Task</Label>
        <Combobox
          value={s.task}
          onValueChange={handleTaskChange}
          options={prompts}
          className="w-full text-xs"
        />
      </div>

      {showCustomPrompt && (
        <div className="flex flex-col gap-1.5">
          <Label className="text-xs">Prompt</Label>
          <Textarea
            value={s.customPrompt}
            onChange={(e) => set({ customPrompt: e.target.value })}
            placeholder="Enter your question or instruction for the model"
            className="min-h-12 text-xs"
            rows={2}
          />
        </div>
      )}

      <ParamSection title="System Prompt" defaultOpen={false}>
        <Textarea
          value={s.system}
          onChange={(e) => set({ system: e.target.value })}
          placeholder="System prompt"
          className="min-h-12 text-xs"
          rows={2}
        />
      </ParamSection>

      <ParamSection title="Advanced Options" defaultOpen={false}>
        <div className="flex flex-col gap-3">
          <SliderField label="Max Tokens" value={s.maxTokens} min={16} max={4096} step={1}
            onChange={(v) => set({ maxTokens: v })} />
          <SliderField label="Num Beams" value={s.numBeams} min={1} max={16} step={1}
            onChange={(v) => set({ numBeams: v })} />
          <SliderField label="Temperature" value={s.temperature} min={0} max={1} step={0.01}
            onChange={(v) => set({ temperature: v })} />
          <SliderField label="Top-K" value={s.topK} min={0} max={99} step={1}
            onChange={(v) => set({ topK: v })} />
          <SliderField label="Top-P" value={s.topP} min={0} max={1} step={0.01}
            onChange={(v) => set({ topP: v })} />

          <SwitchField label="Use Samplers" checked={s.doSample}
            onChange={(v) => set({ doSample: v })} />
          <SwitchField label="Thinking Mode" checked={s.thinkingMode}
            onChange={(v) => set({ thinkingMode: v })} />
          <SwitchField label="Keep Thinking Trace" checked={s.keepThinking}
            onChange={(v) => set({ keepThinking: v })} />
          <SwitchField label="Keep Prefill" checked={s.keepPrefill}
            onChange={(v) => set({ keepPrefill: v })} />

          {s.keepPrefill && (
            <div className="flex flex-col gap-1.5">
              <Label className="text-xs">Prefill Text</Label>
              <Textarea
                value={s.prefill}
                onChange={(e) => set({ prefill: e.target.value })}
                placeholder="Text to prefill the model response"
                className="min-h-12 text-xs"
                rows={2}
              />
            </div>
          )}
        </div>
      </ParamSection>
    </div>
  );
}

function SliderField({ label, value, min, max, step, onChange }: {
  label: string; value: number; min: number; max: number; step: number;
  onChange: (v: number) => void;
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center justify-between">
        <Label className="text-xs">{label}</Label>
        <span className="text-3xs text-muted-foreground tabular-nums">{value}</span>
      </div>
      <Slider value={[value]} min={min} max={max} step={step}
        onValueChange={([v]) => onChange(v)} />
    </div>
  );
}

function SwitchField({ label, checked, onChange }: {
  label: string; checked: boolean; onChange: (v: boolean) => void;
}) {
  return (
    <div className="flex items-center justify-between">
      <Label className="text-xs">{label}</Label>
      <Switch size="sm" checked={checked} onCheckedChange={onChange} />
    </div>
  );
}
