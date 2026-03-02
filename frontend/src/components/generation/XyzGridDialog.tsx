import { useState, useCallback, useMemo } from "react";
import { toast } from "sonner";
import { Loader2, AlertTriangle, Plus, Minus } from "lucide-react";
import { useXyzAxisOptions, useXyzAxisChoices, type XyzAxisOption } from "@/api/hooks/useXyzAxisOptions";
import { useSubmitJob } from "@/api/hooks/useJobs";
import { useJobQueueStore, type JobSnapshot } from "@/stores/jobStore";
import { putJobPayload } from "@/lib/jobPayloadDb";
import { buildXyzScriptArgs, countAxisValues, groupAxisOptions } from "@/lib/xyzGrid";
import type { JobRequest } from "@/api/types/v2";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { Badge } from "@/components/ui/badge";
import { Combobox, type ComboboxGroup } from "@/components/ui/combobox";
import { XyzValueTags } from "./XyzValueTags";

interface XyzGridDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  buildRequest: () => Promise<{ payload: JobRequest; snapshot: JobSnapshot }>;
}

function AxisSection({
  label,
  axisType,
  onAxisTypeChange,
  values,
  onValuesChange,
  groups,
  allOptions,
}: {
  label: string;
  axisType: string;
  onAxisTypeChange: (v: string) => void;
  values: string;
  onValuesChange: (v: string) => void;
  groups: ComboboxGroup[];
  allOptions: XyzAxisOption[];
}) {
  const selectedOption = allOptions.find((o) => o.label === axisType);
  const isStr = selectedOption && (selectedOption.type === "str" || selectedOption.type === "str_permutations");
  const hasChoices = selectedOption && selectedOption.choices === true;

  const { data: choicesData } = useXyzAxisChoices(axisType, !!(isStr && hasChoices));

  const expandedChoices = useMemo(() => {
    if (!choicesData) return [];
    const match = choicesData.find((o) => o.label === axisType);
    return match && Array.isArray(match.choices) ? match.choices : [];
  }, [choicesData, axisType]);

  const selectedTags = useMemo(() => {
    if (!values.trim()) return [];
    return values.split(",").map((s) => s.trim()).filter(Boolean);
  }, [values]);

  const handleTagChange = useCallback((tags: string[]) => {
    onValuesChange(tags.join(", "));
  }, [onValuesChange]);

  const valCount = axisType ? countAxisValues(values, selectedOption?.type ?? "str") : 0;

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <Label className="text-xs font-medium w-6 shrink-0">{label}</Label>
        <div className="flex-1">
          <Combobox
            value={axisType}
            onValueChange={onAxisTypeChange}
            groups={groups}
            placeholder="None (disabled)"
            searchPlaceholder="Search axes..."
            className="h-8"
          />
        </div>
        {axisType && valCount > 0 && (
          <Badge variant="secondary" className="text-3xs tabular-nums shrink-0">{valCount}</Badge>
        )}
      </div>
      {axisType && (
        <div className="pl-8">
          {isStr && hasChoices && expandedChoices.length > 0 ? (
            <XyzValueTags choices={expandedChoices} selected={selectedTags} onChange={handleTagChange} />
          ) : (
            <Input
              value={values}
              onChange={(e) => onValuesChange(e.target.value)}
              placeholder={selectedOption?.type === "int" || selectedOption?.type === "float" ? "10, 20, 30 or 10-50:5" : "value1, value2, value3"}
              className="h-8 text-2xs"
            />
          )}
        </div>
      )}
    </div>
  );
}

export function XyzGridDialog({ open, onOpenChange, buildRequest }: XyzGridDialogProps) {
  const [xType, setXType] = useState("");
  const [yType, setYType] = useState("");
  const [zType, setZType] = useState("");
  const [xValues, setXValues] = useState("");
  const [yValues, setYValues] = useState("");
  const [zValues, setZValues] = useState("");
  const [showZ, setShowZ] = useState(false);
  const [drawLegend, setDrawLegend] = useState(true);
  const [includeImages, setIncludeImages] = useState(true);
  const [includeSubgrids, setIncludeSubgrids] = useState(false);
  const [includeTime, setIncludeTime] = useState(false);
  const [showOptions, setShowOptions] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const { data: axisOptions } = useXyzAxisOptions();
  const submitJob = useSubmitJob();
  const trackJob = useJobQueueStore((s) => s.trackJob);

  const groups = useMemo(() => (axisOptions ? groupAxisOptions(axisOptions) : []), [axisOptions]);

  const allOptions = axisOptions ?? [];
  const findOpt = (label: string) => allOptions.find((o) => o.label === label);

  const xCount = xType ? countAxisValues(xValues, findOpt(xType)?.type ?? "str") : 1;
  const yCount = yType ? countAxisValues(yValues, findOpt(yType)?.type ?? "str") : 1;
  const zCount = showZ && zType ? countAxisValues(zValues, findOpt(zType)?.type ?? "str") : 1;
  const totalCells = Math.max(1, xCount) * Math.max(1, yCount) * Math.max(1, zCount);

  const dimensionText = useMemo(() => {
    const parts: string[] = [];
    if (xType && xCount > 0) parts.push(String(xCount));
    if (yType && yCount > 0) parts.push(String(yCount));
    if (showZ && zType && zCount > 0) parts.push(String(zCount));
    if (parts.length === 0) return null;
    return parts.join(" × ");
  }, [xType, yType, zType, showZ, xCount, yCount, zCount]);

  const canSubmit = (xType && xValues.trim()) || (yType && yValues.trim()) || (showZ && zType && zValues.trim());

  const handleSubmit = useCallback(async () => {
    if (!axisOptions) return;
    setIsSubmitting(true);
    try {
      const { payload, snapshot } = await buildRequest();
      const config = {
        x: { type: xType, values: xValues },
        y: { type: yType, values: yValues },
        z: { type: showZ ? zType : "", values: showZ ? zValues : "" },
        drawLegend,
        includeGrid: true,
        includeSubgrids,
        includeImages,
        includeTime,
        includeText: false,
        marginSize: 0,
      };
      const scriptArgs = buildXyzScriptArgs(config, axisOptions);
      const xyzPayload = { ...payload, script_name: "XYZ Grid Script", script_args: scriptArgs } as JobRequest;
      const job = await submitJob.mutateAsync(xyzPayload);
      const priority = (xyzPayload as { priority?: number }).priority ?? 0;
      trackJob(job.id, "generate", snapshot, xyzPayload, priority);
      putJobPayload({ id: job.id, domain: "generate", request: xyzPayload, priority, snapshot: { controlUnits: snapshot.controlUnits }, createdAt: Date.now() });
      toast.success("XYZ Grid queued", { description: dimensionText ? `${dimensionText} = ${totalCells} images` : "Grid submitted" });
      onOpenChange(false);
    } catch (err) {
      toast.error("Failed to submit XYZ Grid", { description: err instanceof Error ? err.message : String(err) });
    } finally {
      setIsSubmitting(false);
    }
  }, [axisOptions, buildRequest, xType, xValues, yType, yValues, zType, zValues, showZ, drawLegend, includeSubgrids, includeImages, includeTime, submitJob, trackJob, onOpenChange, dimensionText, totalCells]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>XYZ Grid</DialogTitle>
          <DialogDescription>Sweep parameters across X, Y, and Z axes in a single grid job.</DialogDescription>
        </DialogHeader>

        <div className="space-y-3 py-2">
          <AxisSection label="X" axisType={xType} onAxisTypeChange={setXType} values={xValues} onValuesChange={setXValues} groups={groups} allOptions={allOptions} />
          <AxisSection label="Y" axisType={yType} onAxisTypeChange={setYType} values={yValues} onValuesChange={setYValues} groups={groups} allOptions={allOptions} />

          {showZ ? (
            <>
              <AxisSection label="Z" axisType={zType} onAxisTypeChange={setZType} values={zValues} onValuesChange={setZValues} groups={groups} allOptions={allOptions} />
              <button type="button" className="flex items-center gap-1 text-2xs text-muted-foreground hover:text-foreground" onClick={() => { setShowZ(false); setZType(""); setZValues(""); }}>
                <Minus size={12} /> Remove Z axis
              </button>
            </>
          ) : (
            <button type="button" className="flex items-center gap-1 text-2xs text-muted-foreground hover:text-foreground" onClick={() => setShowZ(true)}>
              <Plus size={12} /> Add Z axis
            </button>
          )}

          {/* Grid preview */}
          {dimensionText && (
            <div className="flex items-center gap-2 rounded-md bg-muted/50 px-3 py-2 text-xs">
              <span className="font-medium tabular-nums">{dimensionText}</span>
              <span className="text-muted-foreground">=</span>
              <span className="tabular-nums">{totalCells} image{totalCells !== 1 ? "s" : ""}</span>
              {totalCells > 50 && (
                <span className="ml-auto flex items-center gap-1 text-amber-500">
                  <AlertTriangle size={12} /> Large grid
                </span>
              )}
            </div>
          )}

          {/* Options */}
          <div>
            <button type="button" className="text-2xs text-muted-foreground hover:text-foreground" onClick={() => setShowOptions(!showOptions)}>
              {showOptions ? "Hide options" : "Options..."}
            </button>
            {showOptions && (
              <div className="mt-2 space-y-2 pl-1">
                <label className="flex items-center gap-2 text-2xs">
                  <Checkbox checked={drawLegend} onCheckedChange={(v) => setDrawLegend(!!v)} />
                  Draw legend
                </label>
                <label className="flex items-center gap-2 text-2xs">
                  <Checkbox checked={includeImages} onCheckedChange={(v) => setIncludeImages(!!v)} />
                  Include individual images
                </label>
                {showZ && (
                  <label className="flex items-center gap-2 text-2xs">
                    <Checkbox checked={includeSubgrids} onCheckedChange={(v) => setIncludeSubgrids(!!v)} />
                    Include sub-grids
                  </label>
                )}
                <label className="flex items-center gap-2 text-2xs">
                  <Checkbox checked={includeTime} onCheckedChange={(v) => setIncludeTime(!!v)} />
                  Show timing info
                </label>
              </div>
            )}
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" size="sm" onClick={() => onOpenChange(false)}>Cancel</Button>
          <Button size="sm" onClick={handleSubmit} disabled={isSubmitting || !canSubmit}>
            {isSubmitting && <Loader2 size={14} className="animate-spin mr-1" />}
            Generate Grid
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
