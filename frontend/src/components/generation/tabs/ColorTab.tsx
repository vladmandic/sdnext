import { useCallback, useMemo, useRef } from "react";
import { useGenerationStore } from "@/stores/generationStore";
import { useShallow } from "zustand/react/shallow";
import { ParamSlider } from "../ParamSlider";
import { ParamSection } from "../ParamSection";
import { ParamGrid } from "../ParamRow";
import { ParamLabel } from "../ParamLabel";
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
          <ParamLabel className="text-2xs text-muted-foreground w-16 flex-shrink-0" tooltip="Match output colors to the input image. Active for img2img, inpainting, and edit pipelines.">Enabled</ParamLabel>
          <Switch checked={state.colorCorrectionEnabled} onCheckedChange={set.colorCorrectionEnabled} />
        </div>
        <div className="flex items-center gap-2">
          <ParamLabel className="text-2xs text-muted-foreground w-16 flex-shrink-0" tooltip="Algorithm used to transfer colors from the input image. Histogram matches color distributions channel by channel. Wavelet preserves fine detail by transferring only low-frequency color. AdaIN normalizes mean and variance per channel for a style-transfer effect.">Method</ParamLabel>
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
          <ParamLabel className="text-2xs text-muted-foreground w-16 flex-shrink-0" tooltip="Controls how brightness and color adjustments are applied in latent space. Relative shifts values around their current mean. Absolute sets values to a fixed offset.">Mode</ParamLabel>
          <Combobox
            value={String(state.hdrMode)}
            onValueChange={set.hdrMode}
            options={[{ value: "0", label: "Relative values" }, { value: "1", label: "Absolute values" }]}
            className="h-6 text-2xs flex-1"
          />
        </div>

        <ParamGrid>
          <ParamSlider label="Brightness" value={state.hdrBrightness} onChange={set.hdrBrightness} min={-1} max={1} step={0.1} tooltip="Adjusts the luminance channel in latent space during the final denoising steps. Positive values brighten, negative values darken. Applied before the image is decoded." />
          <ParamSlider label="Sharpen" value={state.hdrSharpen} onChange={set.hdrSharpen} min={-1} max={1} step={0.1} tooltip="Sharpens or softens the latent during late denoising steps. Positive values increase edge contrast, negative values blur. Operates directly on the latent tensor." />
        </ParamGrid>
        <ParamSlider label="Color" value={state.hdrColor} onChange={set.hdrColor} min={0} max={4} step={0.1} tooltip="Centers color channels in latent space during mid-stage denoising. Higher values pull each channel more strongly toward its mean, reducing color drift and improving vibrancy." />

        <div className="flex items-center gap-2">
          <ParamLabel className="text-2xs text-muted-foreground w-16 flex-shrink-0" tooltip="Prunes extreme latent values early in denoising to reduce artifacts at high guidance scales. Outliers beyond the Range and Threshold are pulled back toward the distribution mean.">Clamp</ParamLabel>
          <Switch checked={state.hdrClamp} onCheckedChange={set.hdrClamp} />
        </div>
        <ParamGrid>
          <ParamSlider label="Range" value={state.hdrBoundary} onChange={set.hdrBoundary} min={0} max={10} step={0.1} disabled={!state.hdrClamp} tooltip="Sets the boundary within which latent values are considered normal. Values beyond this range are candidates for clamping. Higher values allow a wider distribution." />
          <ParamSlider label="Threshold" value={state.hdrThreshold} onChange={set.hdrThreshold} min={0} max={1} step={0.01} disabled={!state.hdrClamp} tooltip="Determines which latent values get clamped. Lower values clamp more aggressively, higher values only clamp extreme outliers. Expressed as a fraction of the boundary range." />
        </ParamGrid>

        <div className="flex items-center gap-2">
          <ParamLabel className="text-2xs text-muted-foreground w-16 flex-shrink-0" tooltip="Normalizes the latent tensor to fill the full dynamic range in the final denoising steps. Useful for maximizing contrast before decoding, especially for external post-processing.">Maximize</ParamLabel>
          <Switch checked={state.hdrMaximize} onCheckedChange={set.hdrMaximize} />
        </div>
        <ParamGrid>
          <ParamSlider label="Center" value={state.hdrMaxCenter} onChange={set.hdrMaxCenter} min={0} max={2} step={0.1} disabled={!state.hdrMaximize} tooltip="Controls how strongly each channel is centered before maximizing. Higher values shift channels more toward their mean, balancing the color distribution." />
          <ParamSlider label="Max range" value={state.hdrMaxBoundary} onChange={set.hdrMaxBoundary} min={0.5} max={2} step={0.1} disabled={!state.hdrMaximize} tooltip="Target range multiplier for the normalization. Values above 1.0 stretch the dynamic range beyond default, values below 1.0 compress it." />
        </ParamGrid>

        <ColorPicker label="Tint color" value={state.hdrColorPicker} onChange={set.hdrColorPicker} tooltip="Pick a color to tint the latent during mid-stage denoising. The color is encoded into latent space via TAESD and blended according to Tint strength." />
        <ParamSlider label="Tint strength" value={state.hdrTintRatio} onChange={set.hdrTintRatio} min={-1} max={1} step={0.05} tooltip="Controls how strongly the selected tint color is applied to the latent. Positive values shift toward the tint, negative values shift away from it. 0 disables the tint." />
      </ParamSection>

      <ParamSection title="Basic" defaultOpen={false}>
        <ParamGrid>
          <ParamSlider label="Brightness" value={state.gradingBrightness} onChange={set.gradingBrightness} min={-1} max={1} step={0.05} tooltip="Adjusts overall image brightness after generation. Positive values lighten, negative values darken. Applied as a pixel-level adjustment on the decoded image." />
          <ParamSlider label="Contrast" value={state.gradingContrast} onChange={set.gradingContrast} min={-1} max={1} step={0.05} tooltip="Adjusts the difference between light and dark areas. Positive values increase contrast, negative values flatten the tonal range. 0 leaves the image unchanged." />
          <ParamSlider label="Saturation" value={state.gradingSaturation} onChange={set.gradingSaturation} min={-1} max={1} step={0.05} tooltip="Adjusts color intensity. Positive values make colors more vivid, negative values desaturate toward grayscale. 0 leaves the image unchanged." />
          <ParamSlider label="Hue" value={state.gradingHue} onChange={set.gradingHue} min={0} max={1} step={0.05} tooltip="Rotates the entire color spectrum. 0 and 1 leave colors unchanged, 0.5 shifts all hues by 180 degrees (complementary colors). Useful for creative color shifts." />
          <ParamSlider label="Gamma" value={state.gradingGamma} onChange={set.gradingGamma} min={0.1} max={10} step={0.1} tooltip="Applies a non-linear brightness curve. Values below 1.0 brighten midtones and shadows, values above 1.0 darken them. Default is 1.0 (no change)." />
          <ParamSlider label="Sharpness" value={state.gradingSharpness} onChange={set.gradingSharpness} min={0} max={2} step={0.05} tooltip="Enhances edge definition in the final image using an unsharp mask filter. Higher values produce crisper detail. 0 disables sharpening." />
        </ParamGrid>
        <ParamSlider label="Color temp (K)" value={state.gradingColorTemp} onChange={set.gradingColorTemp} min={2000} max={12000} step={100} tooltip="Shifts the color temperature of the image in Kelvin. Lower values (2000K) produce warm, golden tones. Higher values (12000K) produce cool, blue tones. 6500K is neutral daylight." />
      </ParamSection>

      <ParamSection title="Tone" defaultOpen={false}>
        <ParamGrid>
          <ParamSlider label="Shadows" value={state.gradingShadows} onChange={set.gradingShadows} min={-1} max={1} step={0.05} tooltip="Lightens or darkens the shadow regions of the image. Positive values lift shadows to reveal detail, negative values deepen them. Operates in Lab color space." />
          <ParamSlider label="Midtones" value={state.gradingMidtones} onChange={set.gradingMidtones} min={-1} max={1} step={0.05} tooltip="Adjusts the brightness of midtone values without affecting deep shadows or bright highlights. Useful for fine-tuning overall image exposure." />
          <ParamSlider label="CLAHE clip" value={state.gradingClaheClip} onChange={set.gradingClaheClip} min={0} max={40} step={1} tooltip="Clip limit for Contrast Limited Adaptive Histogram Equalization (CLAHE). Higher values allow more local contrast enhancement. 0 disables CLAHE entirely." />
          <ParamSlider label="CLAHE grid" value={state.gradingClaheGrid} onChange={set.gradingClaheGrid} min={2} max={16} step={1} tooltip="Grid size for CLAHE tile regions. Smaller grids (2-4) produce more localized contrast enhancement, larger grids (8-16) produce a more global effect." />
        </ParamGrid>
        <ParamSlider label="Highlights" value={state.gradingHighlights} onChange={set.gradingHighlights} min={-1} max={1} step={0.05} tooltip="Adjusts the brightness of highlight regions. Positive values brighten highlights, negative values pull them back to recover blown-out detail. Operates in Lab color space." />
      </ParamSection>

      <ParamSection title="Split Toning" defaultOpen={false}>
        <ColorPicker label="Shadows tint" value={state.gradingShadowsTint} onChange={set.gradingShadowsTint} tooltip="Color blended into the dark regions of the image. Set to black (#000000) to disable shadow tinting." />
        <ColorPicker label="Highlights tint" value={state.gradingHighlightsTint} onChange={set.gradingHighlightsTint} tooltip="Color blended into the bright regions of the image. Set to white (#ffffff) to disable highlight tinting." />
        <ParamSlider label="Balance" value={state.gradingSplitToneBalance} onChange={set.gradingSplitToneBalance} min={0} max={1} step={0.05} tooltip="Controls the crossover point between shadow and highlight tinting. 0 shifts the effect entirely toward shadows, 1 shifts it entirely toward highlights. 0.5 is an even split." />
      </ParamSection>

      <ParamSection title="Effects" defaultOpen={false}>
        <ParamGrid>
          <ParamSlider label="Vignette" value={state.gradingVignette} onChange={set.gradingVignette} min={0} max={1} step={0.05} tooltip="Adds radial edge darkening that draws the eye toward the center of the image. Higher values produce a more pronounced dark border. 0 disables the effect." />
          <ParamSlider label="Grain" value={state.gradingGrain} onChange={set.gradingGrain} min={0} max={1} step={0.05} tooltip="Adds random film grain noise to the final image. Higher values produce more visible texture. 0 disables the effect." />
        </ParamGrid>
      </ParamSection>

      <ParamSection title="LUT" defaultOpen={false}>
        <div className="flex items-center gap-2">
          <ParamLabel className="text-2xs text-muted-foreground w-16 flex-shrink-0" tooltip="Upload a .cube LUT (Look-Up Table) file to apply cinematic or stylized color grading. The LUT is applied as the final step after all other color adjustments.">File</ParamLabel>
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
        <ParamSlider label="Strength" value={state.gradingLutStrength} onChange={set.gradingLutStrength} min={0} max={2} step={0.05} disabled={!hasLut} tooltip="Controls the intensity of the LUT color grading. 1.0 applies the LUT at full strength. Values below 1.0 blend with the original colors, values above 1.0 amplify the effect." />
      </ParamSection>
    </div>
  );
}
