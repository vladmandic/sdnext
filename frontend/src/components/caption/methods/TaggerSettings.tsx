import { useState } from "react";
import { Label } from "@/components/ui/label";
import { Combobox } from "@/components/ui/combobox";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { ParamSection } from "@/components/generation/ParamSection";
import { TAGGER_MODELS, TAGGER_DEFAULT } from "@/lib/captionModels";
import type { TaggerOptions } from "@/api/types/caption";

export interface TaggerSettingsValues {
  model: string;
  options: TaggerOptions;
}

interface TaggerSettingsProps {
  onChange: (values: TaggerSettingsValues) => void;
}

export function TaggerSettings({ onChange }: TaggerSettingsProps) {
  const [model, setModel] = useState(TAGGER_DEFAULT);
  const [threshold, setThreshold] = useState(0.5);
  const [characterThreshold, setCharacterThreshold] = useState(0.85);
  const [maxTags, setMaxTags] = useState(74);
  const [includeRating, setIncludeRating] = useState(false);
  const [sortAlpha, setSortAlpha] = useState(false);
  const [useSpaces, setUseSpaces] = useState(false);
  const [escapeBrackets, setEscapeBrackets] = useState(true);
  const [excludeTags, setExcludeTags] = useState("");
  const [showScores, setShowScores] = useState(false);

  const buildOptions = (overrides?: Partial<{
    model: string; threshold: number; characterThreshold: number; maxTags: number;
    includeRating: boolean; sortAlpha: boolean; useSpaces: boolean;
    escapeBrackets: boolean; excludeTags: string; showScores: boolean;
  }>): TaggerOptions => ({
    waifudiffusion_model: overrides?.model ?? model,
    tagger_threshold: overrides?.threshold ?? threshold,
    waifudiffusion_character_threshold: overrides?.characterThreshold ?? characterThreshold,
    tagger_max_tags: overrides?.maxTags ?? maxTags,
    tagger_include_rating: overrides?.includeRating ?? includeRating,
    tagger_sort_alpha: overrides?.sortAlpha ?? sortAlpha,
    tagger_use_spaces: overrides?.useSpaces ?? useSpaces,
    tagger_escape_brackets: overrides?.escapeBrackets ?? escapeBrackets,
    tagger_exclude_tags: overrides?.excludeTags ?? excludeTags,
    tagger_show_scores: overrides?.showScores ?? showScores,
  });

  const emit = (m: string, opts?: TaggerOptions) => {
    onChange({ model: m, options: opts ?? buildOptions() });
  };

  const handleModelChange = (v: string) => {
    setModel(v);
    const opts = buildOptions({ model: v });
    emit(v, opts);
  };

  const emitWithNewOptions = (key: string, value: number | boolean | string) => {
    const opts = buildOptions({ [key]: value });
    emit(model, opts);
  };

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-col gap-1.5">
        <Label className="text-xs">Model</Label>
        <Combobox
          value={model}
          onValueChange={handleModelChange}
          options={TAGGER_MODELS}
          className="w-full text-xs"
        />
      </div>

      <ParamSection title="Advanced Options" defaultOpen={false}>
        <div className="flex flex-col gap-3">
          <SliderField label="General Threshold" value={threshold} min={0} max={1} step={0.01}
            onChange={(v) => { setThreshold(v); emitWithNewOptions("threshold", v); }} />
          <SliderField label="Character Threshold" value={characterThreshold} min={0} max={1} step={0.01}
            onChange={(v) => { setCharacterThreshold(v); emitWithNewOptions("characterThreshold", v); }} />
          <SliderField label="Max Tags" value={maxTags} min={1} max={512} step={1}
            onChange={(v) => { setMaxTags(v); emitWithNewOptions("maxTags", v); }} />

          <SwitchField label="Include Rating" checked={includeRating}
            onChange={(v) => { setIncludeRating(v); emitWithNewOptions("includeRating", v); }} />
          <SwitchField label="Sort Alphabetically" checked={sortAlpha}
            onChange={(v) => { setSortAlpha(v); emitWithNewOptions("sortAlpha", v); }} />
          <SwitchField label="Use Spaces" checked={useSpaces}
            onChange={(v) => { setUseSpaces(v); emitWithNewOptions("useSpaces", v); }} />
          <SwitchField label="Escape Brackets" checked={escapeBrackets}
            onChange={(v) => { setEscapeBrackets(v); emitWithNewOptions("escapeBrackets", v); }} />
          <SwitchField label="Show Confidence Scores" checked={showScores}
            onChange={(v) => { setShowScores(v); emitWithNewOptions("showScores", v); }} />

          <div className="flex flex-col gap-1.5">
            <Label className="text-xs">Exclude Tags</Label>
            <Textarea
              value={excludeTags}
              onChange={(e) => { setExcludeTags(e.target.value); emitWithNewOptions("excludeTags", e.target.value); }}
              placeholder="Comma-separated tags to exclude"
              className="min-h-12 text-xs"
              rows={2}
            />
          </div>
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
