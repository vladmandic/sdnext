import { useMemo, useCallback } from "react";
import { ArrowRight } from "lucide-react";
import { toast } from "sonner";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useGenerationStore, type GenerationResult } from "@/stores/generationStore";
import { extractParamsFromResult } from "@/lib/requestBuilder";
import type { GenerationState } from "@/stores/generationStore";

interface GenerationDiffDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  result: GenerationResult | null;
}

interface DiffGroup {
  label: string;
  keys: (keyof GenerationState)[];
}

const DIFF_GROUPS: DiffGroup[] = [
  { label: "Prompt", keys: ["prompt", "negativePrompt"] },
  { label: "Core", keys: ["sampler", "steps", "width", "height", "batchSize", "batchCount"] },
  { label: "Guidance", keys: ["cfgScale", "cfgEnd", "guidanceRescale", "imageCfgScale", "pagScale", "pagAdaptive", "seed", "subseed", "subseedStrength", "denoisingStrength"] },
  { label: "Hires", keys: ["hiresEnabled", "hiresUpscaler", "hiresScale", "hiresSteps", "hiresDenoising", "hiresResizeMode", "hiresSampler", "hiresForce", "hiresResizeX", "hiresResizeY", "hiresResizeContext"] },
  { label: "Refiner", keys: ["refinerStart", "refinerSteps", "refinerPrompt", "refinerNegative"] },
  { label: "Scheduler", keys: ["sigmaMethod", "timestepSpacing", "betaSchedule", "predictionMethod", "flowShift", "baseShift", "maxShift", "sigmaAdjust", "sigmaAdjustStart", "sigmaAdjustEnd", "thresholding", "dynamic", "rescale", "lowOrder", "timestepsOverride", "timestepsPreset"] },
  { label: "Advanced", keys: ["clipSkip", "vaeType", "tiling", "hidiffusion", "freeuEnabled", "freeuB1", "freeuB2", "freeuS1", "freeuS2", "hypertileUnetEnabled", "hypertileHiresOnly", "hypertileUnetTile", "hypertileUnetMinTile", "hypertileUnetSwapSize", "hypertileUnetDepth", "hypertileVaeEnabled", "hypertileVaeTile", "hypertileVaeSwapSize", "teacacheEnabled", "teacacheThresh", "tokenMergingMethod", "tomeRatio", "todoRatio"] },
  { label: "Detailer", keys: ["detailerEnabled", "detailerModels", "detailerPrompt", "detailerNegative", "detailerSteps", "detailerStrength", "detailerResolution", "detailerMaxDetected", "detailerPadding", "detailerBlur", "detailerConfidence", "detailerIou", "detailerMinSize", "detailerMaxSize", "detailerRenoise", "detailerRenoiseEnd", "detailerSegmentation", "detailerIncludeDetections", "detailerMerge", "detailerSort", "detailerClasses"] },
  { label: "Latent Corrections", keys: ["hdrMode", "hdrBrightness", "hdrSharpen", "hdrColor", "hdrClamp", "hdrBoundary", "hdrThreshold", "hdrMaximize", "hdrMaxCenter", "hdrMaxBoundary", "hdrColorPicker", "hdrTintRatio"] },
  { label: "Color Grading", keys: ["gradingBrightness", "gradingContrast", "gradingSaturation", "gradingHue", "gradingGamma", "gradingSharpness", "gradingColorTemp", "gradingShadows", "gradingMidtones", "gradingHighlights", "gradingClaheClip", "gradingClaheGrid", "gradingShadowsTint", "gradingHighlightsTint", "gradingSplitToneBalance", "gradingVignette", "gradingGrain", "gradingLutFile", "gradingLutStrength"] },
];

interface DiffRow {
  key: string;
  current: unknown;
  result: unknown;
  changed: boolean;
}

function formatValue(v: unknown): string {
  if (v === undefined || v === null) return "-";
  if (typeof v === "boolean") return v ? "Yes" : "No";
  if (typeof v === "string") return v || '""';
  if (Array.isArray(v)) return v.join(", ");
  return String(v);
}

export function GenerationDiffDialog({ open, onOpenChange, result }: GenerationDiffDialogProps) {
  const storeState = useGenerationStore();
  const setParams = useGenerationStore((s) => s.setParams);

  const resultParams = useMemo(
    () => (result ? extractParamsFromResult(result) : {}),
    [result],
  );

  const groupedRows = useMemo(() => {
    const current = storeState as unknown as Record<string, unknown>;
    const mapped = resultParams as Record<string, unknown>;

    return DIFF_GROUPS.map((group) => {
      const rows: DiffRow[] = group.keys
        .filter((k) => k in mapped)
        .map((k) => {
          const cur = current[k];
          const res = mapped[k];
          return { key: k, current: cur, result: res, changed: JSON.stringify(cur) !== JSON.stringify(res) };
        });
      const changedCount = rows.filter((r) => r.changed).length;
      return { ...group, rows, changedCount };
    }).filter((g) => g.changedCount > 0);
  }, [storeState, resultParams]);

  const totalChanged = groupedRows.reduce((sum, g) => sum + g.changedCount, 0);

  const handleApplyAll = useCallback(() => {
    const updates: Record<string, unknown> = {};
    for (const group of groupedRows) {
      for (const row of group.rows) {
        if (row.changed) updates[row.key] = row.result;
      }
    }
    setParams(updates);
    toast.success(`Applied ${totalChanged} changed parameter${totalChanged !== 1 ? "s" : ""}`);
    onOpenChange(false);
  }, [groupedRows, totalChanged, setParams, onOpenChange]);

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
          <DialogTitle>Compare Settings ({totalChanged} changed)</DialogTitle>
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
              {groupedRows.map((group) => (
                <GroupSection
                  key={group.label}
                  label={group.label}
                  changedCount={group.changedCount}
                  rows={group.rows}
                  onApplyOne={handleApplyOne}
                />
              ))}
            </tbody>
          </table>
        </ScrollArea>
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>Close</Button>
          <Button onClick={handleApplyAll} disabled={totalChanged === 0}>Apply All ({totalChanged})</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function GroupSection({ label, changedCount, rows, onApplyOne }: {
  label: string;
  changedCount: number;
  rows: DiffRow[];
  onApplyOne: (key: string, value: unknown) => void;
}) {
  return (
    <>
      <tr className="border-t border-border/50">
        <td colSpan={4} className="py-1 px-2 font-semibold text-muted-foreground text-3xs uppercase tracking-wider">
          {label} ({changedCount})
        </td>
      </tr>
      {rows.map((row) => (
        <tr key={row.key} className={row.changed ? "bg-amber-500/10" : "text-muted-foreground/60"}>
          <td className="py-0.5 px-2 font-mono">{row.key}</td>
          <td className="py-0.5 px-2 max-w-32 truncate">{formatValue(row.current)}</td>
          <td className="py-0.5 px-2 max-w-32 truncate font-medium">{formatValue(row.result)}</td>
          <td className="py-0.5 px-1">
            {row.changed && (
              <button type="button" onClick={() => onApplyOne(row.key, row.result)} className="hover:text-primary" title="Apply this value">
                <ArrowRight size={12} />
              </button>
            )}
          </td>
        </tr>
      ))}
    </>
  );
}
