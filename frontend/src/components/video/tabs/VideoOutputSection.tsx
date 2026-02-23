import { useVideoStore } from "@/stores/videoStore";
import { ParamSection } from "@/components/generation/ParamSection";
import { ParamSlider } from "@/components/generation/ParamSlider";
import { ParamGrid } from "@/components/generation/ParamRow";
import { Combobox } from "@/components/ui/combobox";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";

export function VideoOutputSection() {
  const fps = useVideoStore((s) => s.fps);
  const interpolate = useVideoStore((s) => s.interpolate);
  const codec = useVideoStore((s) => s.codec);
  const format = useVideoStore((s) => s.format);
  const codecOptions = useVideoStore((s) => s.codecOptions);
  const saveVideo = useVideoStore((s) => s.saveVideo);
  const saveFrames = useVideoStore((s) => s.saveFrames);
  const saveSafetensors = useVideoStore((s) => s.saveSafetensors);
  const setParam = useVideoStore((s) => s.setParam);

  return (
    <ParamSection title="Output" defaultOpen={false}>
      <ParamGrid>
        <ParamSlider label="FPS" value={fps} onChange={(v) => setParam("fps", v)} min={1} max={60} step={1} />
        <ParamSlider label="Interpolate" value={interpolate} onChange={(v) => setParam("interpolate", v)} min={0} max={8} step={1} />
      </ParamGrid>
      <div className="flex items-center gap-2">
        <Label className="text-2xs text-muted-foreground w-16 shrink-0">Codec</Label>
        <Combobox value={codec} onValueChange={(v) => setParam("codec", v)} options={["libx264", "libx265", "libvpx-vp9", "libaom-av1"]} className="h-6 text-2xs flex-1" />
      </div>
      <div className="flex items-center gap-2">
        <Label className="text-2xs text-muted-foreground w-16 shrink-0">Format</Label>
        <Combobox value={format} onValueChange={(v) => setParam("format", v)} options={["mp4", "webm", "gif"]} className="h-6 text-2xs flex-1" />
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
