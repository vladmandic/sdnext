import { useGenerationStore } from "@/stores/generationStore";
import { ParamSlider } from "../ParamSlider";
import { ParamSection } from "../ParamSection";

export function GuidanceTab() {
  const store = useGenerationStore();

  return (
    <div className="flex flex-col gap-3 text-sm">
      <ParamSection title="Guidance">
        <ParamSlider label="Guidance scale" value={store.cfgScale} onChange={(v) => store.setParam("cfgScale", v)} min={0} max={30} step={0.5} />
        <ParamSlider label="Guidance end" value={store.cfgEnd} onChange={(v) => store.setParam("cfgEnd", v)} min={0} max={1} step={0.1} />
        <ParamSlider label="Rescale" value={store.guidanceRescale} onChange={(v) => store.setParam("guidanceRescale", v)} min={0} max={1} step={0.05} />
      </ParamSection>

      <ParamSection title="Refine Guidance" defaultOpen={false}>
        <ParamSlider label="Image CFG" value={store.imageCfgScale} onChange={(v) => store.setParam("imageCfgScale", v)} min={0} max={30} step={0.1} />
      </ParamSection>

      <ParamSection title="Attention Guidance" defaultOpen={false}>
        <ParamSlider label="PAG scale" value={store.pagScale} onChange={(v) => store.setParam("pagScale", v)} min={0} max={30} step={0.05} />
        <ParamSlider label="Adaptive" value={store.pagAdaptive} onChange={(v) => store.setParam("pagAdaptive", v)} min={0} max={1} step={0.05} />
      </ParamSection>

      <ParamSection title="Denoising" defaultOpen={false}>
        <ParamSlider label="Strength" value={store.denoisingStrength} onChange={(v) => store.setParam("denoisingStrength", v)} min={0} max={1} step={0.05} />
      </ParamSection>
    </div>
  );
}
