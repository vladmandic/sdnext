import { useCallback, useEffect, useMemo } from "react";
import { Play, Loader2 } from "lucide-react";
import { toast } from "sonner";
import { useProcessStore } from "@/stores/processStore";
import { useUpscalerList } from "@/api/hooks/useModels";
import { useSubmitJob } from "@/api/hooks/useJobs";
import { useJobWebSocket } from "@/api/hooks/useJobWebSocket";
import { uploadFile } from "@/lib/upload";
import { ParamSlider } from "@/components/generation/ParamSlider";
import { Combobox } from "@/components/ui/combobox";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { api } from "@/api/client";

export function ProcessPanel() {
  const image = useProcessStore((s) => s.image);
  const upscaler = useProcessStore((s) => s.upscaler);
  const scale = useProcessStore((s) => s.scale);
  const isProcessing = useProcessStore((s) => s.isProcessing);
  const jobId = useProcessStore((s) => s.jobId);
  const setUpscaler = useProcessStore((s) => s.setUpscaler);
  const setScale = useProcessStore((s) => s.setScale);
  const setProcessing = useProcessStore((s) => s.setProcessing);
  const setJobId = useProcessStore((s) => s.setJobId);
  const setResult = useProcessStore((s) => s.setResult);

  const { data: upscalers } = useUpscalerList();
  const submitJob = useSubmitJob();
  const { result: wsResult, status: wsStatus, error: wsError } = useJobWebSocket(jobId);

  const upscalerNames = useMemo(() => upscalers?.map((u) => u.name) ?? [], [upscalers]);

  // Auto-select first non-"None" upscaler
  useEffect(() => {
    if (upscaler === "None" && upscalerNames.length > 0) {
      const first = upscalerNames.find((n) => n !== "None");
      if (first) setUpscaler(first);
    }
  }, [upscalerNames, upscaler, setUpscaler]);

  // Handle job completion
  useEffect(() => {
    if (!jobId) return;
    if (wsStatus === "completed" && wsResult) {
      const img = wsResult.images[0];
      if (img) {
        setResult(`${api.getBaseUrl()}${img.url}`, img.width, img.height);
      }
      setJobId(null);
      setProcessing(false);
    } else if (wsStatus === "failed") {
      toast.error("Upscale failed", { description: wsError ?? "Unknown error" });
      setJobId(null);
      setProcessing(false);
    } else if (wsStatus === "cancelled") {
      setJobId(null);
      setProcessing(false);
    }
  }, [wsStatus, wsResult, wsError, jobId, setResult, setJobId, setProcessing]);

  const handleProcess = useCallback(async () => {
    if (!image || isProcessing) return;
    setProcessing(true);
    setResult(null);
    try {
      const ref = await uploadFile(image);
      const job = await submitJob.mutateAsync({ type: "upscale", image: ref, upscaler, scale });
      setJobId(job.id);
    } catch (err) {
      toast.error("Upscale failed", { description: err instanceof Error ? err.message : String(err) });
      setProcessing(false);
    }
  }, [image, isProcessing, upscaler, scale, setProcessing, setResult, submitJob, setJobId]);

  return (
    <div className="flex flex-col h-full">
      <div className="px-3 pt-3 pb-2 border-b border-border space-y-3">
        <div className="space-y-1.5">
          <Label className="text-[11px] text-muted-foreground">Upscaler</Label>
          <Combobox
            value={upscaler}
            onValueChange={setUpscaler}
            options={upscalerNames}
            placeholder="Select upscaler..."
            className="h-7 text-xs"
          />
        </div>
        <ParamSlider label="Scale" value={scale} onChange={setScale} min={1} max={8} step={0.5} />
        <Button
          type="button"
          onClick={handleProcess}
          disabled={!image || isProcessing || upscaler === "None"}
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
