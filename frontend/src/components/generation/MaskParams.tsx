import { useCallback } from "react";
import { useImg2ImgStore } from "@/stores/img2imgStore";
import { ParamSection } from "@/components/generation/ParamSection";
import { ParamSlider } from "@/components/generation/ParamSlider";
import { Switch } from "@/components/ui/switch";
import { ParamLabel } from "./ParamLabel";

export function MaskParams() {
  const maskLineCount = useImg2ImgStore((s) => s.maskLines.length);
  const maskBlur = useImg2ImgStore((s) => s.maskBlur);
  const inpaintFullRes = useImg2ImgStore((s) => s.inpaintFullRes);
  const inpaintFullResPadding = useImg2ImgStore((s) => s.inpaintFullResPadding);
  const inpaintingMaskInvert = useImg2ImgStore((s) => s.inpaintingMaskInvert);
  const setMaskBlur = useImg2ImgStore((s) => s.setMaskBlur);
  const setInpaintFullRes = useImg2ImgStore((s) => s.setInpaintFullRes);
  const setInpaintFullResPadding = useImg2ImgStore((s) => s.setInpaintFullResPadding);
  const setInpaintingMaskInvert = useImg2ImgStore((s) => s.setInpaintingMaskInvert);

  const handleFullResToggle = useCallback((checked: boolean) => setInpaintFullRes(checked), [setInpaintFullRes]);
  const handleInvertToggle = useCallback((checked: boolean) => setInpaintingMaskInvert(checked), [setInpaintingMaskInvert]);

  if (maskLineCount === 0) return null;

  return (
    <ParamSection title="Mask" defaultOpen>
      <ParamSlider label="Blur" value={maskBlur} onChange={setMaskBlur} min={0} max={64} step={1} />

      <div className="flex items-center justify-between">
        <ParamLabel className="text-2xs text-muted-foreground">Inpaint full res</ParamLabel>
        <Switch checked={inpaintFullRes} onCheckedChange={handleFullResToggle} />
      </div>

      {inpaintFullRes && (
        <ParamSlider label="Padding" value={inpaintFullResPadding} onChange={setInpaintFullResPadding} min={0} max={256} step={4} />
      )}

      <div className="flex items-center justify-between">
        <ParamLabel className="text-2xs text-muted-foreground">Invert mask</ParamLabel>
        <Switch checked={inpaintingMaskInvert} onCheckedChange={handleInvertToggle} />
      </div>
    </ParamSection>
  );
}
