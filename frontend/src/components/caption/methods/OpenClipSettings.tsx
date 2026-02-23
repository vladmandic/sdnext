import { Label } from "@/components/ui/label";
import { Combobox } from "@/components/ui/combobox";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { ParamSection } from "@/components/generation/ParamSection";
import { useCaptionSettingsStore } from "@/stores/captionSettingsStore";
import { useOpenClipModels } from "@/api/hooks/useCaption";
import { INTERROGATE_MODES, BLIP_MODEL_NAMES } from "@/lib/captionModels";

export function OpenClipSettings() {
  const s = useCaptionSettingsStore((st) => st.openclip);
  const set = useCaptionSettingsStore((st) => st.setOpenClip);

  const { data: clipModels } = useOpenClipModels();
  const clipModelList = clipModels ?? [];

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-col gap-1.5">
        <Label className="text-xs">CLIP Model</Label>
        <Combobox
          value={s.clipModel}
          onValueChange={(v) => set({ clipModel: v })}
          options={clipModelList}
          placeholder={clipModelList.length === 0 ? "Loading..." : "Select model"}
          className="w-full text-xs"
        />
      </div>

      <div className="flex flex-col gap-1.5">
        <Label className="text-xs">Caption Model</Label>
        <Combobox
          value={s.blipModel}
          onValueChange={(v) => set({ blipModel: v })}
          options={BLIP_MODEL_NAMES}
          className="w-full text-xs"
        />
      </div>

      <div className="flex flex-col gap-1.5">
        <Label className="text-xs">Mode</Label>
        <Combobox
          value={s.mode}
          onValueChange={(v) => set({ mode: v })}
          options={INTERROGATE_MODES}
          className="w-full text-xs"
        />
      </div>

      <div className="flex items-center justify-between">
        <Label className="text-xs">Analyze</Label>
        <Switch size="sm" checked={s.analyze} onCheckedChange={(v) => set({ analyze: v })} />
      </div>

      <ParamSection title="Advanced Options" defaultOpen={false}>
        <div className="flex flex-col gap-3">
          <SliderField label="Max Length" value={s.maxLength} min={1} max={512} step={1}
            onChange={(v) => set({ maxLength: v })} />
          <SliderField label="Chunk Size" value={s.chunkSize} min={256} max={4096} step={64}
            onChange={(v) => set({ chunkSize: v })} />
          <SliderField label="Min Flavors" value={s.minFlavors} min={0} max={32} step={1}
            onChange={(v) => set({ minFlavors: v })} />
          <SliderField label="Max Flavors" value={s.maxFlavors} min={0} max={32} step={1}
            onChange={(v) => set({ maxFlavors: v })} />
          <SliderField label="Intermediates" value={s.flavorCount} min={256} max={4096} step={64}
            onChange={(v) => set({ flavorCount: v })} />
          <SliderField label="Num Beams" value={s.numBeams} min={1} max={16} step={1}
            onChange={(v) => set({ numBeams: v })} />
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
