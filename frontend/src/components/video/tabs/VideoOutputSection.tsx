import { useMemo } from "react";
import { useVideoStore } from "@/stores/videoStore";
import { ParamSection } from "@/components/generation/ParamSection";
import { ParamSlider } from "@/components/generation/ParamSlider";
import { ParamGrid } from "@/components/generation/ParamRow";
import { Combobox } from "@/components/ui/combobox";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { OUTPUT_PRESETS, qualityToCrf, crfToQuality } from "@/lib/videoOutputPresets";

const presetOptions = OUTPUT_PRESETS.map((p) => p.id);
const presetLabels: Record<string, string> = Object.fromEntries(OUTPUT_PRESETS.map((p) => [p.id, p.label]));

export function VideoOutputSection() {
  const fps = useVideoStore((s) => s.fps);
  const interpolate = useVideoStore((s) => s.interpolate);
  const codec = useVideoStore((s) => s.codec);
  const format = useVideoStore((s) => s.format);
  const codecOptions = useVideoStore((s) => s.codecOptions);
  const outputPreset = useVideoStore((s) => s.outputPreset);
  const outputQuality = useVideoStore((s) => s.outputQuality);
  const saveVideo = useVideoStore((s) => s.saveVideo);
  const saveFrames = useVideoStore((s) => s.saveFrames);
  const saveSafetensors = useVideoStore((s) => s.saveSafetensors);
  const width = useVideoStore((s) => s.width);
  const fpResolution = useVideoStore((s) => s.fpResolution);
  const setParam = useVideoStore((s) => s.setParam);

  const isCustom = outputPreset === "custom";
  const isLossless = outputPreset === "lossless";
  const showQuality = !isCustom && !isLossless;

  const resolutionHint = useMemo(() => {
    const maxRes = Math.max(width, fpResolution);
    return maxRes > 1080 && outputQuality < 50;
  }, [width, fpResolution, outputQuality]);

  const handlePresetChange = (id: string) => {
    const preset = OUTPUT_PRESETS.find((p) => p.id === id);
    if (!preset) return;
    setParam("outputPreset", id);
    if (id !== "custom") {
      setParam("codec", preset.codec);
      setParam("format", preset.format);
      if (preset.codecOptions) {
        setParam("codecOptions", preset.codecOptions);
        setParam("outputQuality", crfToQuality(preset.codecOptions));
      } else {
        setParam("codecOptions", "");
      }
    }
  };

  const handleQualityChange = (value: number) => {
    setParam("outputQuality", value);
    const currentOpts = codecOptions;
    const hasExtraOpts = currentOpts.replace(/crf:\d+/, "").replace(/^,|,$/g, "").trim();
    const crfStr = qualityToCrf(value);
    setParam("codecOptions", hasExtraOpts ? `${crfStr},${hasExtraOpts}` : crfStr);
  };

  const presetDescription = OUTPUT_PRESETS.find((p) => p.id === outputPreset)?.description;

  return (
    <ParamSection title="Output" defaultOpen={false}>
      <div className="flex items-center gap-2">
        <Label className="text-2xs text-muted-foreground w-16 shrink-0">Preset</Label>
        <Combobox
          value={outputPreset}
          onValueChange={handlePresetChange}
          options={presetOptions}
          renderLabel={(_v, label) => presetLabels[_v] ?? label}
          className="h-6 text-2xs flex-1"
        />
      </div>
      {presetDescription && (
        <p className="text-3xs text-muted-foreground ml-18 pl-0.5">{presetDescription}</p>
      )}

      {showQuality && (
        <ParamSlider label="Quality" value={outputQuality} onChange={handleQualityChange} min={10} max={100} step={5} />
      )}
      {resolutionHint && showQuality && (
        <p className="text-3xs text-amber-500 ml-18 pl-0.5">Consider higher quality for this resolution</p>
      )}

      {isCustom && (
        <>
          <div className="flex items-center gap-2">
            <Label className="text-2xs text-muted-foreground w-16 shrink-0">Codec</Label>
            <Combobox value={codec} onValueChange={(v) => setParam("codec", v)} options={["libx264", "libx265", "libvpx-vp9", "libaom-av1", "ffv1"]} className="h-6 text-2xs flex-1" />
          </div>
          <div className="flex items-center gap-2">
            <Label className="text-2xs text-muted-foreground w-16 shrink-0">Format</Label>
            <Combobox value={format} onValueChange={(v) => setParam("format", v)} options={["mp4", "webm", "mkv", "gif"]} className="h-6 text-2xs flex-1" />
          </div>
          <div className="flex items-center gap-2">
            <Label className="text-2xs text-muted-foreground w-16 shrink-0">Codec opts</Label>
            <input
              type="text"
              value={codecOptions}
              onChange={(e) => setParam("codecOptions", e.target.value)}
              className="flex-1 h-6 text-2xs px-2 rounded border border-input bg-background"
            />
          </div>
        </>
      )}

      <ParamGrid>
        <ParamSlider label="FPS" value={fps} onChange={(v) => setParam("fps", v)} min={1} max={60} step={1} />
        <ParamSlider label="Interpolate" value={interpolate} onChange={(v) => setParam("interpolate", v)} min={0} max={8} step={1} />
      </ParamGrid>

      <div className="flex items-center gap-2">
        <Label className="text-2xs text-muted-foreground w-16 shrink-0">Video</Label>
        <Switch checked={saveVideo} onCheckedChange={(v) => setParam("saveVideo", v)} />
      </div>
      <div className="flex items-center gap-2">
        <Label className="text-2xs text-muted-foreground w-16 shrink-0">Frames</Label>
        <Switch checked={saveFrames} onCheckedChange={(v) => setParam("saveFrames", v)} />
      </div>
      <div className="flex items-center gap-2">
        <Label className="text-2xs text-muted-foreground w-16 shrink-0">Safetensors</Label>
        <Switch checked={saveSafetensors} onCheckedChange={(v) => setParam("saveSafetensors", v)} />
      </div>
    </ParamSection>
  );
}
