import { useMemo } from "react";
import { useGenerationStore } from "@/stores/generationStore";
import { useShallow } from "zustand/react/shallow";
import { ParamSlider } from "../ParamSlider";
import { ParamSection } from "../ParamSection";
import { ParamGrid } from "../ParamRow";
import { getParamHelp } from "@/data/parameterHelp";

export function GuidanceTab() {
  const state = useGenerationStore(useShallow((s) => ({
    cfgScale: s.cfgScale,
    cfgEnd: s.cfgEnd,
    guidanceRescale: s.guidanceRescale,
    imageCfgScale: s.imageCfgScale,
    pagScale: s.pagScale,
    pagAdaptive: s.pagAdaptive,
  })));
  const setParam = useGenerationStore((s) => s.setParam);

  const set = useMemo(() => ({
    cfgScale: (v: number) => setParam("cfgScale", v),
    cfgEnd: (v: number) => setParam("cfgEnd", v),
    guidanceRescale: (v: number) => setParam("guidanceRescale", v),
    imageCfgScale: (v: number) => setParam("imageCfgScale", v),
    pagScale: (v: number) => setParam("pagScale", v),
    pagAdaptive: (v: number) => setParam("pagAdaptive", v),
  }), [setParam]);

  return (
    <div className="flex flex-col gap-3 text-sm">
      <ParamSection title="Guidance">
        <ParamGrid>
          <ParamSlider label="Guidance scale" value={state.cfgScale} onChange={set.cfgScale} min={0} max={30} step={0.5} />
          <ParamSlider label="Guidance end" value={state.cfgEnd} onChange={set.cfgEnd} min={0} max={1} step={0.1} />
        </ParamGrid>
        <ParamSlider label="Rescale" tooltip={getParamHelp("guidance rescale")} value={state.guidanceRescale} onChange={set.guidanceRescale} min={0} max={1} step={0.05} />
      </ParamSection>

      <ParamSection title="Refine Guidance" defaultOpen={false}>
        <ParamSlider label="Refine guidance scale" value={state.imageCfgScale} onChange={set.imageCfgScale} min={0} max={30} step={0.1} />
      </ParamSection>

      <ParamSection title="Attention Guidance" defaultOpen={false}>
        <ParamGrid>
          <ParamSlider label="PAG scale" value={state.pagScale} onChange={set.pagScale} min={0} max={30} step={0.05} />
          <ParamSlider label="Adaptive" tooltip={getParamHelp("adaptive scaling")} value={state.pagAdaptive} onChange={set.pagAdaptive} min={0} max={1} step={0.05} />
        </ParamGrid>
      </ParamSection>
    </div>
  );
}
