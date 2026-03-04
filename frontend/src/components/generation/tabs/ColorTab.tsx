import { useCallback, useMemo, useRef } from "react";
import { useGenerationStore } from "@/stores/generationStore";
import { useShallow } from "zustand/react/shallow";
import { ParamSlider } from "../ParamSlider";
import { ParamSection } from "../ParamSection";
import { ParamGrid } from "../ParamRow";
import { ParamLabel } from "../ParamLabel";
import { getParamHelp } from "@/data/parameterHelp";
import { Switch } from "@/components/ui/switch";
import { Combobox } from "@/components/ui/combobox";
import { ColorPicker } from "@/components/ui/color-picker";
import { Button } from "@/components/ui/button";
import { Upload, X } from "lucide-react";
import { uploadFile } from "@/lib/upload";

export function ColorTab() {
  const state = useGenerationStore(useShallow((s) => ({
    colorCorrectionEnabled: s.colorCorrectionEnabled,
    colorCorrectionMethod: s.colorCorrectionMethod,
    hdrMode: s.hdrMode,
    hdrBrightness: s.hdrBrightness,
    hdrSharpen: s.hdrSharpen,
    hdrColor: s.hdrColor,
    hdrClamp: s.hdrClamp,
    hdrBoundary: s.hdrBoundary,
    hdrThreshold: s.hdrThreshold,
    hdrMaximize: s.hdrMaximize,
    hdrMaxCenter: s.hdrMaxCenter,
    hdrMaxBoundary: s.hdrMaxBoundary,
    hdrColorPicker: s.hdrColorPicker,
    hdrTintRatio: s.hdrTintRatio,
    gradingBrightness: s.gradingBrightness,
    gradingContrast: s.gradingContrast,
    gradingSaturation: s.gradingSaturation,
    gradingHue: s.gradingHue,
    gradingGamma: s.gradingGamma,
    gradingSharpness: s.gradingSharpness,
    gradingColorTemp: s.gradingColorTemp,
    gradingShadows: s.gradingShadows,
    gradingMidtones: s.gradingMidtones,
    gradingHighlights: s.gradingHighlights,
    gradingClaheClip: s.gradingClaheClip,
    gradingClaheGrid: s.gradingClaheGrid,
    gradingShadowsTint: s.gradingShadowsTint,
    gradingHighlightsTint: s.gradingHighlightsTint,
    gradingSplitToneBalance: s.gradingSplitToneBalance,
    gradingVignette: s.gradingVignette,
    gradingGrain: s.gradingGrain,
    gradingLutFile: s.gradingLutFile,
    gradingLutStrength: s.gradingLutStrength,
  })));
  const setParam = useGenerationStore((s) => s.setParam);

  const set = useMemo(() => ({
    colorCorrectionEnabled: (checked: boolean) => setParam("colorCorrectionEnabled", checked),
    colorCorrectionMethod: (v: string) => setParam("colorCorrectionMethod", v),
    hdrMode: (v: string) => setParam("hdrMode", Number(v)),
    hdrBrightness: (v: number) => setParam("hdrBrightness", v),
    hdrSharpen: (v: number) => setParam("hdrSharpen", v),
    hdrColor: (v: number) => setParam("hdrColor", v),
    hdrClamp: (checked: boolean) => setParam("hdrClamp", checked),
    hdrBoundary: (v: number) => setParam("hdrBoundary", v),
    hdrThreshold: (v: number) => setParam("hdrThreshold", v),
    hdrMaximize: (checked: boolean) => setParam("hdrMaximize", checked),
    hdrMaxCenter: (v: number) => setParam("hdrMaxCenter", v),
    hdrMaxBoundary: (v: number) => setParam("hdrMaxBoundary", v),
    hdrColorPicker: (v: string) => setParam("hdrColorPicker", v),
    hdrTintRatio: (v: number) => setParam("hdrTintRatio", v),
    gradingBrightness: (v: number) => setParam("gradingBrightness", v),
    gradingContrast: (v: number) => setParam("gradingContrast", v),
    gradingSaturation: (v: number) => setParam("gradingSaturation", v),
    gradingHue: (v: number) => setParam("gradingHue", v),
    gradingGamma: (v: number) => setParam("gradingGamma", v),
    gradingSharpness: (v: number) => setParam("gradingSharpness", v),
    gradingColorTemp: (v: number) => setParam("gradingColorTemp", v),
    gradingShadows: (v: number) => setParam("gradingShadows", v),
    gradingMidtones: (v: number) => setParam("gradingMidtones", v),
    gradingHighlights: (v: number) => setParam("gradingHighlights", v),
    gradingClaheClip: (v: number) => setParam("gradingClaheClip", v),
    gradingClaheGrid: (v: number) => setParam("gradingClaheGrid", v),
    gradingShadowsTint: (v: string) => setParam("gradingShadowsTint", v),
    gradingHighlightsTint: (v: string) => setParam("gradingHighlightsTint", v),
    gradingSplitToneBalance: (v: number) => setParam("gradingSplitToneBalance", v),
    gradingVignette: (v: number) => setParam("gradingVignette", v),
    gradingGrain: (v: number) => setParam("gradingGrain", v),
    gradingLutStrength: (v: number) => setParam("gradingLutStrength", v),
  }), [setParam]);

  const lutInputRef = useRef<HTMLInputElement>(null);
  const lutFileName = state.gradingLutFile ? state.gradingLutFile.split("/").pop() ?? "" : "";
  const hasLut = !!state.gradingLutFile;

  const handleLutFile = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    const ref = await uploadFile(file);
    setParam("gradingLutFile", ref);
  }, [setParam]);

  const clearLut = useCallback(() => {
    setParam("gradingLutFile", "");
  }, [setParam]);

  return (
    <div className="flex flex-col gap-3 text-sm">
      <ParamSection title="Color Correction" defaultOpen={false}>
        <div className="flex items-center gap-2">
          <ParamLabel className="text-2xs text-muted-foreground w-16 flex-shrink-0" tooltip={getParamHelp("color correction enabled")}>Enabled</ParamLabel>
          <Switch checked={state.colorCorrectionEnabled} onCheckedChange={set.colorCorrectionEnabled} />
        </div>
        <div data-param="method" className="flex items-center gap-2">
          <ParamLabel className="text-2xs text-muted-foreground w-16 flex-shrink-0" tooltip={getParamHelp("color correction method")}>Method</ParamLabel>
          <Combobox
            value={state.colorCorrectionMethod}
            onValueChange={set.colorCorrectionMethod}
            options={[{ value: "histogram", label: "Histogram" }, { value: "wavelet", label: "Wavelet" }, { value: "adain", label: "AdaIN" }]}
            className="h-6 text-2xs flex-1"
            disabled={!state.colorCorrectionEnabled}
          />
        </div>
      </ParamSection>

      <ParamSection title="Latent Corrections" defaultOpen={false}>
        <div className="flex items-center gap-2">
          <ParamLabel className="text-2xs text-muted-foreground w-16 flex-shrink-0" tooltip={getParamHelp("latent mode")}>Mode</ParamLabel>
          <Combobox
            value={String(state.hdrMode)}
            onValueChange={set.hdrMode}
            options={[{ value: "0", label: "Relative values" }, { value: "1", label: "Absolute values" }]}
            className="h-6 text-2xs flex-1"
          />
        </div>

        <ParamGrid>
          <ParamSlider label="Brightness" value={state.hdrBrightness} onChange={set.hdrBrightness} min={-1} max={1} step={0.1} tooltip={getParamHelp("latent brightness")} />
          <ParamSlider label="Sharpen" value={state.hdrSharpen} onChange={set.hdrSharpen} min={-1} max={1} step={0.1} tooltip={getParamHelp("latent sharpen")} />
        </ParamGrid>
        <ParamSlider label="Color" value={state.hdrColor} onChange={set.hdrColor} min={0} max={4} step={0.1} tooltip={getParamHelp("latent color")} />

        <div className="flex items-center gap-2">
          <ParamLabel className="text-2xs text-muted-foreground w-16 flex-shrink-0" tooltip={getParamHelp("hdr clamp")}>Clamp</ParamLabel>
          <Switch checked={state.hdrClamp} onCheckedChange={set.hdrClamp} />
        </div>
        <ParamGrid>
          <ParamSlider label="Range" value={state.hdrBoundary} onChange={set.hdrBoundary} min={0} max={10} step={0.1} disabled={!state.hdrClamp} tooltip={getParamHelp("latent clamp range")} />
          <ParamSlider label="Threshold" value={state.hdrThreshold} onChange={set.hdrThreshold} min={0} max={1} step={0.01} disabled={!state.hdrClamp} tooltip={getParamHelp("latent clamp threshold")} />
        </ParamGrid>

        <div className="flex items-center gap-2">
          <ParamLabel className="text-2xs text-muted-foreground w-16 flex-shrink-0" tooltip={getParamHelp("hdr maximize")}>Maximize</ParamLabel>
          <Switch checked={state.hdrMaximize} onCheckedChange={set.hdrMaximize} />
        </div>
        <ParamGrid>
          <ParamSlider label="Center" value={state.hdrMaxCenter} onChange={set.hdrMaxCenter} min={0} max={2} step={0.1} disabled={!state.hdrMaximize} tooltip={getParamHelp("hdr maximize center")} />
          <ParamSlider label="Max range" value={state.hdrMaxBoundary} onChange={set.hdrMaxBoundary} min={0.5} max={2} step={0.1} disabled={!state.hdrMaximize} tooltip={getParamHelp("hdr maximize range")} />
        </ParamGrid>

        <ColorPicker label="Tint color" value={state.hdrColorPicker} onChange={set.hdrColorPicker} />
        <ParamSlider label="Tint strength" value={state.hdrTintRatio} onChange={set.hdrTintRatio} min={-1} max={1} step={0.05} />
      </ParamSection>

      <ParamSection title="Basic" defaultOpen={false}>
        <ParamGrid>
          <ParamSlider label="Brightness" value={state.gradingBrightness} onChange={set.gradingBrightness} min={-1} max={1} step={0.05} tooltip={getParamHelp("grading brightness")} />
          <ParamSlider label="Contrast" value={state.gradingContrast} onChange={set.gradingContrast} min={-1} max={1} step={0.05} tooltip={getParamHelp("grading contrast")} />
          <ParamSlider label="Saturation" value={state.gradingSaturation} onChange={set.gradingSaturation} min={-1} max={1} step={0.05} tooltip={getParamHelp("grading saturation")} />
          <ParamSlider label="Hue" value={state.gradingHue} onChange={set.gradingHue} min={0} max={1} step={0.05} tooltip={getParamHelp("grading hue")} />
          <ParamSlider label="Gamma" value={state.gradingGamma} onChange={set.gradingGamma} min={0.1} max={10} step={0.1} tooltip={getParamHelp("grading gamma")} />
          <ParamSlider label="Sharpness" value={state.gradingSharpness} onChange={set.gradingSharpness} min={0} max={2} step={0.05} tooltip={getParamHelp("grading sharpness")} />
        </ParamGrid>
        <ParamSlider label="Color temp (K)" value={state.gradingColorTemp} onChange={set.gradingColorTemp} min={2000} max={12000} step={100} />
      </ParamSection>

      <ParamSection title="Tone" defaultOpen={false}>
        <ParamGrid>
          <ParamSlider label="Shadows" value={state.gradingShadows} onChange={set.gradingShadows} min={-1} max={1} step={0.05} tooltip={getParamHelp("grading shadows")} />
          <ParamSlider label="Midtones" value={state.gradingMidtones} onChange={set.gradingMidtones} min={-1} max={1} step={0.05} tooltip={getParamHelp("grading midtones")} />
          <ParamSlider label="CLAHE clip" value={state.gradingClaheClip} onChange={set.gradingClaheClip} min={0} max={40} step={1} />
          <ParamSlider label="CLAHE grid" value={state.gradingClaheGrid} onChange={set.gradingClaheGrid} min={2} max={16} step={1} />
        </ParamGrid>
        <ParamSlider label="Highlights" value={state.gradingHighlights} onChange={set.gradingHighlights} min={-1} max={1} step={0.05} tooltip={getParamHelp("grading highlights")} />
      </ParamSection>

      <ParamSection title="Split Toning" defaultOpen={false}>
        <ColorPicker label="Shadows tint" value={state.gradingShadowsTint} onChange={set.gradingShadowsTint} />
        <ColorPicker label="Highlights tint" value={state.gradingHighlightsTint} onChange={set.gradingHighlightsTint} />
        <ParamSlider label="Balance" value={state.gradingSplitToneBalance} onChange={set.gradingSplitToneBalance} min={0} max={1} step={0.05} tooltip={getParamHelp("split tone balance")} />
      </ParamSection>

      <ParamSection title="Effects" defaultOpen={false}>
        <ParamGrid>
          <ParamSlider label="Vignette" value={state.gradingVignette} onChange={set.gradingVignette} min={0} max={1} step={0.05} tooltip={getParamHelp("grading vignette")} />
          <ParamSlider label="Grain" value={state.gradingGrain} onChange={set.gradingGrain} min={0} max={1} step={0.05} tooltip={getParamHelp("grading grain")} />
        </ParamGrid>
      </ParamSection>

      <ParamSection title="LUT" defaultOpen={false}>
        <div className="flex items-center gap-2">
          <ParamLabel className="text-2xs text-muted-foreground w-16 flex-shrink-0" tooltip={getParamHelp("lut file")}>File</ParamLabel>
          {hasLut ? (
            <div className="flex items-center gap-1.5 flex-1 min-w-0">
              <span className="text-2xs truncate flex-1">{lutFileName}</span>
              <Button variant="ghost" size="icon-sm" onClick={clearLut} title="Remove LUT">
                <X size={12} />
              </Button>
            </div>
          ) : (
            <Button variant="outline" size="sm" className="h-6 text-2xs gap-1.5 flex-1" onClick={() => lutInputRef.current?.click()}>
              <Upload size={12} />
              Upload .cube file
            </Button>
          )}
          <input ref={lutInputRef} type="file" accept=".cube" className="hidden" onChange={handleLutFile} />
        </div>
        <ParamSlider label="Strength" value={state.gradingLutStrength} onChange={set.gradingLutStrength} min={0} max={2} step={0.05} disabled={!hasLut} tooltip={getParamHelp("lut strength")} />
      </ParamSection>
    </div>
  );
}
