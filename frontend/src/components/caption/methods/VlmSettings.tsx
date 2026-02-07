import { useState, useMemo } from "react";
import { Label } from "@/components/ui/label";
import { Combobox } from "@/components/ui/combobox";
import { Textarea } from "@/components/ui/textarea";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { ParamSection } from "@/components/generation/ParamSection";
import { VLM_DEFAULT, VLM_MODEL_NAMES, VLM_SYSTEM_DEFAULT, getPromptsForModel, VLM_PROMPT_MAPPING } from "@/lib/captionModels";
import type { VlmOptions } from "@/api/types/caption";

export interface VlmSettingsValues {
  model: string;
  question: string;
  system: string;
  options: VlmOptions;
}

interface VlmSettingsProps {
  onChange: (values: VlmSettingsValues) => void;
}

export function VlmSettings({ onChange }: VlmSettingsProps) {
  const [model, setModel] = useState(VLM_DEFAULT);
  const [task, setTask] = useState("Normal Caption");
  const [customPrompt, setCustomPrompt] = useState("");
  const [system, setSystem] = useState(VLM_SYSTEM_DEFAULT);

  // Advanced options
  const [maxTokens, setMaxTokens] = useState(512);
  const [numBeams, setNumBeams] = useState(1);
  const [temperature, setTemperature] = useState(0.8);
  const [topK, setTopK] = useState(0);
  const [topP, setTopP] = useState(0);
  const [useSamplers, setUseSamplers] = useState(true);
  const [thinkingMode, setThinkingMode] = useState(false);
  const [keepThinking, setKeepThinking] = useState(false);
  const [keepPrefill, setKeepPrefill] = useState(false);
  const [prefillText, setPrefillText] = useState("");

  const prompts = useMemo(() => getPromptsForModel(model), [model]);

  const buildOptions = (overrides?: Partial<{
    maxTokens: number; numBeams: number; temperature: number;
    topK: number; topP: number; useSamplers: boolean;
    thinkingMode: boolean; keepThinking: boolean; keepPrefill: boolean;
  }>): VlmOptions => ({
    interrogate_vlm_max_length: overrides?.maxTokens ?? maxTokens,
    interrogate_vlm_num_beams: overrides?.numBeams ?? numBeams,
    interrogate_vlm_temperature: overrides?.temperature ?? temperature,
    interrogate_vlm_top_k: overrides?.topK ?? topK,
    interrogate_vlm_top_p: overrides?.topP ?? topP,
    interrogate_vlm_do_sample: overrides?.useSamplers ?? useSamplers,
    interrogate_vlm_thinking_mode: overrides?.thinkingMode ?? thinkingMode,
    interrogate_vlm_keep_thinking: overrides?.keepThinking ?? keepThinking,
    interrogate_vlm_keep_prefill: overrides?.keepPrefill ?? keepPrefill,
  });

  const emitChange = (m: string, t: string, cp: string, sys: string, opts?: VlmOptions) => {
    const mapped = VLM_PROMPT_MAPPING[t] ?? t;
    const question = t === "Use Prompt" ? cp : mapped;
    onChange({ model: m, question, system: sys, options: opts ?? buildOptions() });
  };

  const handleModelChange = (v: string) => {
    setModel(v);
    const newPrompts = getPromptsForModel(v);
    const newTask = newPrompts.includes(task) ? task : "Normal Caption";
    setTask(newTask);
    emitChange(v, newTask, customPrompt, system);
  };

  const handleTaskChange = (v: string) => {
    setTask(v);
    emitChange(model, v, customPrompt, system);
  };

  const handlePromptChange = (v: string) => {
    setCustomPrompt(v);
    emitChange(model, task, v, system);
  };

  const handleSystemChange = (v: string) => {
    setSystem(v);
    emitChange(model, task, customPrompt, v);
  };

  const emitWithNewOptions = (key: string, value: number | boolean) => {
    const opts = buildOptions({ [key]: value });
    emitChange(model, task, customPrompt, system, opts);
  };

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-col gap-1.5">
        <Label className="text-xs">Model</Label>
        <Combobox
          value={model}
          onValueChange={handleModelChange}
          options={VLM_MODEL_NAMES}
          className="w-full text-xs"
        />
      </div>

      <div className="flex flex-col gap-1.5">
        <Label className="text-xs">Task</Label>
        <Combobox
          value={task}
          onValueChange={handleTaskChange}
          options={prompts}
          className="w-full text-xs"
        />
      </div>

      {task === "Use Prompt" && (
        <div className="flex flex-col gap-1.5">
          <Label className="text-xs">Prompt</Label>
          <Textarea
            value={customPrompt}
            onChange={(e) => handlePromptChange(e.target.value)}
            placeholder="Enter your question or instruction for the model"
            className="min-h-12 text-xs"
            rows={2}
          />
        </div>
      )}

      <ParamSection title="System Prompt" defaultOpen={false}>
        <Textarea
          value={system}
          onChange={(e) => handleSystemChange(e.target.value)}
          placeholder="System prompt"
          className="min-h-12 text-xs"
          rows={2}
        />
      </ParamSection>

      <ParamSection title="Advanced Options" defaultOpen={false}>
        <div className="flex flex-col gap-3">
          <SliderField label="Max Tokens" value={maxTokens} min={16} max={4096} step={1}
            onChange={(v) => { setMaxTokens(v); emitWithNewOptions("maxTokens", v); }} />
          <SliderField label="Num Beams" value={numBeams} min={1} max={16} step={1}
            onChange={(v) => { setNumBeams(v); emitWithNewOptions("numBeams", v); }} />
          <SliderField label="Temperature" value={temperature} min={0} max={1} step={0.01}
            onChange={(v) => { setTemperature(v); emitWithNewOptions("temperature", v); }} />
          <SliderField label="Top-K" value={topK} min={0} max={99} step={1}
            onChange={(v) => { setTopK(v); emitWithNewOptions("topK", v); }} />
          <SliderField label="Top-P" value={topP} min={0} max={1} step={0.01}
            onChange={(v) => { setTopP(v); emitWithNewOptions("topP", v); }} />

          <SwitchField label="Use Samplers" checked={useSamplers}
            onChange={(v) => { setUseSamplers(v); emitWithNewOptions("useSamplers", v); }} />
          <SwitchField label="Thinking Mode" checked={thinkingMode}
            onChange={(v) => { setThinkingMode(v); emitWithNewOptions("thinkingMode", v); }} />
          <SwitchField label="Keep Thinking Trace" checked={keepThinking}
            onChange={(v) => { setKeepThinking(v); emitWithNewOptions("keepThinking", v); }} />
          <SwitchField label="Keep Prefill" checked={keepPrefill}
            onChange={(v) => { setKeepPrefill(v); emitWithNewOptions("keepPrefill", v); }} />

          {keepPrefill && (
            <div className="flex flex-col gap-1.5">
              <Label className="text-xs">Prefill Text</Label>
              <Textarea
                value={prefillText}
                onChange={(e) => setPrefillText(e.target.value)}
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
        <span className="text-[10px] text-muted-foreground tabular-nums">{value}</span>
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
