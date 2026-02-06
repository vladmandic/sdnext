import { useAdapterStore } from "@/stores/adapterStore";
import { ParamSlider } from "../ParamSlider";
import { ParamSection } from "../ParamSection";
import { AdapterUnit } from "./adapters/AdapterUnit";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";

export function AdaptersTab() {
  const activeUnits = useAdapterStore((s) => s.activeUnits);
  const setActiveUnits = useAdapterStore((s) => s.setActiveUnits);
  const unloadAdapter = useAdapterStore((s) => s.unloadAdapter);

  return (
    <div className="flex flex-col gap-3 text-sm">
      <ParamSection title="IP-Adapter">
        <ParamSlider label="Units" value={activeUnits} onChange={setActiveUnits} min={1} max={4} />

        <div className="flex items-center gap-2">
          <Label className="text-[11px] text-muted-foreground w-16 flex-shrink-0">Unload</Label>
          <Switch checked={unloadAdapter} onCheckedChange={(checked) => useAdapterStore.setState({ unloadAdapter: checked })} />
        </div>
      </ParamSection>

      <Accordion type="multiple" defaultValue={["unit-0"]} className="w-full">
        {Array.from({ length: activeUnits }, (_, i) => (
          <AccordionItem key={i} value={`unit-${i}`}>
            <AccordionTrigger className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground py-2">
              Unit {i + 1}
            </AccordionTrigger>
            <AccordionContent>
              <AdapterUnit index={i} />
            </AccordionContent>
          </AccordionItem>
        ))}
      </Accordion>
    </div>
  );
}
