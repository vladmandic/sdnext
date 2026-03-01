import { useMemo, useCallback } from "react";
import { ArrowRight } from "lucide-react";
import { toast } from "sonner";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useVideoStore } from "@/stores/videoStore";

interface ParamDiffDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  resultParams: Record<string, unknown>;
  domain: string;
}

const SHARED_KEYS = ["engine", "model", "prompt", "negative", "width", "height", "frames", "steps", "sampler", "samplerShift", "dynamicShift", "seed", "guidanceScale", "guidanceTrue", "initStrength", "vaeType", "vaeTileFrames"];

const OUTPUT_KEYS = ["fps", "interpolate", "codec", "format", "codecOptions", "saveVideo", "saveFrames", "saveSafetensors"];

const FP_KEYS = ["fpVariant", "fpResolution", "fpDuration", "fpLatentWindowSize", "fpSteps", "fpShift", "fpCfgScale", "fpCfgDistilled", "fpCfgRescale", "fpStartWeight", "fpEndWeight", "fpVisionWeight", "fpSectionPrompt", "fpSystemPrompt", "fpTeacache", "fpOptimizedPrompt", "fpCfgZero", "fpPreview", "fpAttention", "fpVaeType"];

const LTX_KEYS = ["ltxModel", "ltxSteps", "ltxDecodeTimestep", "ltxNoiseScale", "ltxUpsampleEnable", "ltxUpsampleRatio", "ltxRefineEnable", "ltxRefineStrength", "ltxConditionStrength", "ltxAudioEnable"];

const API_TO_STORE: Record<string, string> = {
  guidance_scale: "guidanceScale", guidance_true: "guidanceTrue", sampler_shift: "samplerShift",
  dynamic_shift: "dynamicShift", init_strength: "initStrength", vae_type: "vaeType", vae_tile_frames: "vaeTileFrames",
  codec_options: "codecOptions", save_video: "saveVideo", save_frames: "saveFrames", save_safetensors: "saveSafetensors",
  fp_variant: "fpVariant", fp_resolution: "fpResolution", fp_duration: "fpDuration",
  fp_latent_window_size: "fpLatentWindowSize", fp_steps: "fpSteps", fp_shift: "fpShift",
  fp_cfg_scale: "fpCfgScale", fp_cfg_distilled: "fpCfgDistilled", fp_cfg_rescale: "fpCfgRescale",
  fp_start_weight: "fpStartWeight", fp_end_weight: "fpEndWeight", fp_vision_weight: "fpVisionWeight",
  fp_section_prompt: "fpSectionPrompt", fp_system_prompt: "fpSystemPrompt", fp_teacache: "fpTeacache",
  fp_optimized_prompt: "fpOptimizedPrompt", fp_cfg_zero: "fpCfgZero", fp_preview: "fpPreview",
  fp_attention: "fpAttention", fp_vae_type: "fpVaeType",
  ltx_model: "ltxModel", ltx_steps: "ltxSteps", ltx_decode_timestep: "ltxDecodeTimestep",
  ltx_noise_scale: "ltxNoiseScale", ltx_upsample_enable: "ltxUpsampleEnable", ltx_upsample_ratio: "ltxUpsampleRatio",
  ltx_refine_enable: "ltxRefineEnable", ltx_refine_strength: "ltxRefineStrength",
  ltx_condition_strength: "ltxConditionStrength", ltx_audio_enable: "ltxAudioEnable",
};

function normalizeResultParams(raw: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(raw)) {
    const storeKey = API_TO_STORE[k] ?? k;
    out[storeKey] = v;
  }
  return out;
}

function formatValue(v: unknown): string {
  if (v === undefined || v === null) return "-";
  if (typeof v === "boolean") return v ? "Yes" : "No";
  if (typeof v === "string") return v || '""';
  return String(v);
}

function keysForDomain(domain: string): string[] {
  const keys = [...SHARED_KEYS, ...OUTPUT_KEYS];
  if (domain === "framepack" || domain === "") keys.push(...FP_KEYS);
  if (domain === "ltx" || domain === "") keys.push(...LTX_KEYS);
  return keys;
}

interface DiffRow {
  key: string;
  current: unknown;
  result: unknown;
  changed: boolean;
}

export function ParamDiffDialog({ open, onOpenChange, resultParams, domain }: ParamDiffDialogProps) {
  const storeState = useVideoStore();
  const setParams = useVideoStore((s) => s.setParams);

  const normalized = useMemo(() => normalizeResultParams(resultParams), [resultParams]);

  const rows = useMemo<DiffRow[]>(() => {
    const keys = keysForDomain(domain);
    return keys
      .filter((k) => k in normalized)
      .map((k) => {
        const current = (storeState as unknown as Record<string, unknown>)[k];
        const result = normalized[k];
        return { key: k, current, result, changed: JSON.stringify(current) !== JSON.stringify(result) };
      });
  }, [storeState, normalized, domain]);

  const changedCount = rows.filter((r) => r.changed).length;

  const handleApplyAll = useCallback(() => {
    const updates: Record<string, unknown> = {};
    for (const row of rows) {
      if (row.changed) updates[row.key] = row.result;
    }
    setParams(updates);
    toast.success(`Applied ${changedCount} changed parameter${changedCount !== 1 ? "s" : ""}`);
    onOpenChange(false);
  }, [rows, changedCount, setParams, onOpenChange]);

  const handleApplyOne = useCallback(
    (key: string, value: unknown) => {
      setParams({ [key]: value });
      toast.success(`Applied ${key}`);
    },
    [setParams],
  );

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-xl">
        <DialogHeader>
          <DialogTitle>Compare Settings ({changedCount} changed)</DialogTitle>
        </DialogHeader>
        <ScrollArea className="max-h-[60vh]">
          <table className="w-full text-2xs">
            <thead>
              <tr className="border-b text-left text-muted-foreground">
                <th className="py-1 px-2 font-medium">Parameter</th>
                <th className="py-1 px-2 font-medium">Current</th>
                <th className="py-1 px-2 font-medium">Result</th>
                <th className="py-1 px-2 w-8" />
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.key} className={row.changed ? "bg-amber-500/10" : "text-muted-foreground/60"}>
                  <td className="py-0.5 px-2 font-mono">{row.key}</td>
                  <td className="py-0.5 px-2 max-w-32 truncate">{formatValue(row.current)}</td>
                  <td className="py-0.5 px-2 max-w-32 truncate font-medium">{formatValue(row.result)}</td>
                  <td className="py-0.5 px-1">
                    {row.changed && (
                      <button type="button" onClick={() => handleApplyOne(row.key, row.result)} className="hover:text-primary" title="Apply this value">
                        <ArrowRight size={12} />
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </ScrollArea>
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>Close</Button>
          <Button onClick={handleApplyAll} disabled={changedCount === 0}>Apply All ({changedCount})</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
