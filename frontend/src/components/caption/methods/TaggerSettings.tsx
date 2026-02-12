import { useMemo } from "react";
import { Label } from "@/components/ui/label";
import { Combobox } from "@/components/ui/combobox";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { ParamSection } from "@/components/generation/ParamSection";
import { useCaptionSettingsStore } from "@/stores/captionSettingsStore";
import { useTaggerModels } from "@/api/hooks/useCaption";
import { TAGGER_DEFAULT } from "@/lib/captionModels";

export function TaggerSettings() {
  const s = useCaptionSettingsStore((st) => st.tagger);
  const set = useCaptionSettingsStore((st) => st.setTagger);

  const { data: models } = useTaggerModels();
  const modelNames = useMemo(() => models?.map((m) => m.name) ?? [], [models]);

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-col gap-1.5">
        <Label className="text-xs">Model</Label>
        <Combobox
          value={s.model}
          onValueChange={(v) => set({ model: v })}
          options={modelNames.length > 0 ? modelNames : [TAGGER_DEFAULT]}
          placeholder={modelNames.length === 0 ? "Loading..." : "Select model"}
          className="w-full text-xs"
        />
      </div>

      <ParamSection title="Advanced Options" defaultOpen={false}>
        <div className="flex flex-col gap-3">
          <SliderField label="General Threshold" value={s.threshold} min={0} max={1} step={0.01}
            onChange={(v) => set({ threshold: v })} />
          <SliderField label="Character Threshold" value={s.characterThreshold} min={0} max={1} step={0.01}
            onChange={(v) => set({ characterThreshold: v })} />
          <SliderField label="Max Tags" value={s.maxTags} min={1} max={512} step={1}
            onChange={(v) => set({ maxTags: v })} />

          <SwitchField label="Include Rating" checked={s.includeRating}
            onChange={(v) => set({ includeRating: v })} />
          <SwitchField label="Sort Alphabetically" checked={s.sortAlpha}
            onChange={(v) => set({ sortAlpha: v })} />
          <SwitchField label="Use Spaces" checked={s.useSpaces}
            onChange={(v) => set({ useSpaces: v })} />
          <SwitchField label="Escape Brackets" checked={s.escapeBrackets}
            onChange={(v) => set({ escapeBrackets: v })} />
          <SwitchField label="Show Confidence Scores" checked={s.showScores}
            onChange={(v) => set({ showScores: v })} />

          <div className="flex flex-col gap-1.5">
            <Label className="text-xs">Exclude Tags</Label>
            <Textarea
              value={s.excludeTags}
              onChange={(e) => set({ excludeTags: e.target.value })}
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
