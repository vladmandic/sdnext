import { useCallback, useEffect, useMemo } from "react";
import { Play, Loader2 } from "lucide-react";
import { useProcessStore } from "@/stores/processStore";
import { useJobQueueStore, selectDomainActive } from "@/stores/jobStore";
import { useUpscalerList, useUpscalerGroups } from "@/api/hooks/useModels";
import { useSubmitToQueue } from "@/hooks/useSubmitToQueue";
import { uploadFile } from "@/lib/upload";
import { ParamSlider } from "@/components/generation/ParamSlider";
import { Combobox } from "@/components/ui/combobox";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";

export function ProcessPanel() {
  const image = useProcessStore((s) => s.image);
  const upscaler = useProcessStore((s) => s.upscaler);
  const scale = useProcessStore((s) => s.scale);
  const setUpscaler = useProcessStore((s) => s.setUpscaler);
  const setScale = useProcessStore((s) => s.setScale);
  const setResult = useProcessStore((s) => s.setResult);

  const isProcessing = useJobQueueStore(selectDomainActive("upscale"));

  const { data: upscalers } = useUpscalerList();
  const upscalerGroups = useUpscalerGroups();

  // Auto-select first non-"None" upscaler
  useEffect(() => {
    if (upscaler === "None" && upscalers && upscalers.length > 0) {
      const first = upscalers.find((u) => u.name !== "None");
      if (first) setUpscaler(first.name);
    }
  }, [upscalers, upscaler, setUpscaler]);

  const buildRequest = useCallback(async () => {
    if (!image) throw new Error("No image selected");
    setResult(null);
    const ref = await uploadFile(image);
    return {
      payload: { type: "upscale" as const, image: ref, upscaler, scale },
      snapshot: {},
    };
  }, [image, upscaler, scale, setResult]);

  const { submit, isSubmitting } = useSubmitToQueue(useMemo(() => ({ domain: "upscale" as const, buildRequest }), [buildRequest]));

  return (
    <div className="flex flex-col h-full">
      <div className="px-3 pt-3 pb-2 border-b border-border space-y-3">
        <div className="space-y-1.5">
          <Label className="text-2xs text-muted-foreground">Upscaler</Label>
          <Combobox
            value={upscaler}
            onValueChange={setUpscaler}
            groups={upscalerGroups}
            placeholder="Select upscaler..."
            className="h-6 text-2xs"
          />
        </div>
        <ParamSlider label="Scale" value={scale} onChange={setScale} min={1} max={8} step={0.5} />
        <Button
          type="button"
          onClick={submit}
          disabled={!image || isProcessing || isSubmitting || upscaler === "None"}
          size="sm"
          className="w-full"
        >
          {isProcessing ? (
            <>
              <Loader2 size={14} className="animate-spin" />
              Processing...
            </>
          ) : (
            <>
              <Play size={14} />
              Process
            </>
          )}
        </Button>
      </div>
    </div>
  );
}
