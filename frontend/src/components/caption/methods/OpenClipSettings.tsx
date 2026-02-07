import { useState, useMemo } from "react";
import { Label } from "@/components/ui/label";
import { Combobox } from "@/components/ui/combobox";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { ParamSection } from "@/components/generation/ParamSection";
import { useInterrogateModels } from "@/api/hooks/useCaption";
import { INTERROGATE_MODES, BLIP_MODEL_NAMES } from "@/lib/captionModels";
import type { OpenClipOptions } from "@/api/types/caption";

export interface OpenClipSettingsValues {
  model: string;
  blipModel: string;
  mode: string;
  analyze: boolean;
  options: OpenClipOptions;
}

interface OpenClipSettingsProps {
  onChange: (values: OpenClipSettingsValues) => void;
}

export function OpenClipSettings({ onChange }: OpenClipSettingsProps) {
  const { data: allModels } = useInterrogateModels();
  const clipModels = useMemo(
    () => (allModels ?? []).filter((m) => m.toLowerCase() !== "deepdanbooru"),
    [allModels],
  );

  const [model, setModel] = useState("");
  const [blipModel, setBlipModel] = useState(BLIP_MODEL_NAMES[0] ?? "blip-base");
  const [mode, setMode] = useState<string>("fast");
  const [analyze, setAnalyze] = useState(false);

  // Advanced options
  const [minLength, setMinLength] = useState(32);
  const [maxLength, setMaxLength] = useState(74);
  const [chunkSize, setChunkSize] = useState(1024);
  const [minFlavors, setMinFlavors] = useState(2);
  const [maxFlavors, setMaxFlavors] = useState(16);
  const [intermediates, setIntermediates] = useState(1024);
  const [clipNumBeams, setClipNumBeams] = useState(1);

  const effectiveModel = model || clipModels[0] || "";

  const buildOptions = (overrides?: Partial<{
    blipModel: string; minLength: number; maxLength: number; chunkSize: number;
    minFlavors: number; maxFlavors: number; intermediates: number; clipNumBeams: number;
  }>): OpenClipOptions => ({
    interrogate_blip_model: overrides?.blipModel ?? blipModel,
    interrogate_clip_num_beams: overrides?.clipNumBeams ?? clipNumBeams,
    interrogate_clip_min_length: overrides?.minLength ?? minLength,
    interrogate_clip_max_length: overrides?.maxLength ?? maxLength,
    interrogate_clip_min_flavors: overrides?.minFlavors ?? minFlavors,
    interrogate_clip_max_flavors: overrides?.maxFlavors ?? maxFlavors,
    interrogate_clip_flavor_count: overrides?.intermediates ?? intermediates,
    interrogate_clip_chunk_size: overrides?.chunkSize ?? chunkSize,
  });

  const emit = (m: string, bm: string, mo: string, a: boolean, opts?: OpenClipOptions) => {
    onChange({ model: m, blipModel: bm, mode: mo, analyze: a, options: opts ?? buildOptions() });
  };

  const handleModelChange = (v: string) => {
    setModel(v);
    emit(v, blipModel, mode, analyze);
  };

  const handleBlipModelChange = (v: string) => {
    setBlipModel(v);
    const opts = buildOptions({ blipModel: v });
    emit(effectiveModel, v, mode, analyze, opts);
  };

  const handleModeChange = (v: string) => {
    setMode(v);
    emit(effectiveModel, blipModel, v, analyze);
  };

  const handleAnalyzeChange = (v: boolean) => {
    setAnalyze(v);
    emit(effectiveModel, blipModel, mode, v);
  };

  const emitWithNewOptions = (key: string, value: number) => {
    const opts = buildOptions({ [key]: value });
    emit(effectiveModel, blipModel, mode, analyze, opts);
  };

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-col gap-1.5">
        <Label className="text-xs">CLIP Model</Label>
        <Combobox
          value={effectiveModel}
          onValueChange={handleModelChange}
          options={clipModels}
          placeholder={clipModels.length === 0 ? "Loading..." : "Select model"}
          className="w-full text-xs"
        />
      </div>

      <div className="flex flex-col gap-1.5">
        <Label className="text-xs">Caption Model</Label>
        <Combobox
          value={blipModel}
          onValueChange={handleBlipModelChange}
          options={BLIP_MODEL_NAMES}
          className="w-full text-xs"
        />
      </div>

      <div className="flex flex-col gap-1.5">
        <Label className="text-xs">Mode</Label>
        <Combobox
          value={mode}
          onValueChange={handleModeChange}
          options={INTERROGATE_MODES}
          className="w-full text-xs"
        />
      </div>

      <div className="flex items-center justify-between">
        <Label className="text-xs">Analyze</Label>
        <Switch size="sm" checked={analyze} onCheckedChange={handleAnalyzeChange} />
      </div>

      <ParamSection title="Advanced Options" defaultOpen={false}>
        <div className="flex flex-col gap-3">
          <SliderField label="Min Length" value={minLength} min={1} max={128} step={1}
            onChange={(v) => { setMinLength(v); emitWithNewOptions("minLength", v); }} />
          <SliderField label="Max Length" value={maxLength} min={1} max={512} step={1}
            onChange={(v) => { setMaxLength(v); emitWithNewOptions("maxLength", v); }} />
          <SliderField label="Chunk Size" value={chunkSize} min={256} max={4096} step={64}
            onChange={(v) => { setChunkSize(v); emitWithNewOptions("chunkSize", v); }} />
          <SliderField label="Min Flavors" value={minFlavors} min={0} max={32} step={1}
            onChange={(v) => { setMinFlavors(v); emitWithNewOptions("minFlavors", v); }} />
          <SliderField label="Max Flavors" value={maxFlavors} min={0} max={32} step={1}
            onChange={(v) => { setMaxFlavors(v); emitWithNewOptions("maxFlavors", v); }} />
          <SliderField label="Intermediates" value={intermediates} min={256} max={4096} step={64}
            onChange={(v) => { setIntermediates(v); emitWithNewOptions("intermediates", v); }} />
          <SliderField label="CLIP Num Beams" value={clipNumBeams} min={1} max={16} step={1}
            onChange={(v) => { setClipNumBeams(v); emitWithNewOptions("clipNumBeams", v); }} />
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
