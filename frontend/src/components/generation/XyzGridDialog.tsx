import { useState, useCallback, useMemo } from "react";
import { toast } from "sonner";
import { Loader2, AlertTriangle, Plus, Minus, CheckCircle2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { useXyzAxisOptions, useXyzAxisChoices, useXyzValidate, useXyzPreview, type XyzAxisOption, type XyzValidateResponse, type XyzPreviewResponse } from "@/api/hooks/useXyzAxisOptions";
import { useSubmitJob } from "@/api/hooks/useJobs";
import { useJobQueueStore, type JobSnapshot } from "@/stores/jobStore";
import { putJobPayload } from "@/lib/jobPayloadDb";
import { countAxisValues, groupAxisOptions } from "@/lib/xyzGrid";
import type { JobRequest, XyzGridJobParams } from "@/api/types/v2";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { Badge } from "@/components/ui/badge";
import { Combobox, type ComboboxGroup } from "@/components/ui/combobox";

interface XyzGridDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  buildRequest: () => Promise<{ payload: JobRequest; snapshot: JobSnapshot }>;
}

interface AxisValidation {
  response?: XyzValidateResponse;
  isLoading: boolean;
}

function AxisSection({
  label,
  axisType,
  onAxisTypeChange,
  values,
  onValuesChange,
  groups,
  allOptions,
  validation,
  onValidate,
}: {
  label: string;
  axisType: string;
  onAxisTypeChange: (v: string) => void;
  values: string;
  onValuesChange: (v: string) => void;
  groups: ComboboxGroup[];
  allOptions: XyzAxisOption[];
  validation?: AxisValidation;
  onValidate: () => void;
}) {
  const [choiceFilter, setChoiceFilter] = useState("");

  const selectedOption = allOptions.find((o) => o.label === axisType);
  const hasChoices = selectedOption?.has_choices ?? false;

  const { data: choicesData } = useXyzAxisChoices(axisType, hasChoices);

  const expandedChoices = useMemo(() => {
    if (!choicesData) return [];
    const match = choicesData.find((o) => o.label === axisType);
    return match?.choices ?? [];
  }, [choicesData, axisType]);

  const filteredChoices = useMemo(() => {
    if (!choiceFilter) return expandedChoices;
    const q = choiceFilter.toLowerCase();
    return expandedChoices.filter((c) => c.toLowerCase().includes(q));
  }, [expandedChoices, choiceFilter]);

  const selectedSet = useMemo(() => {
    if (!values.trim()) return new Set<string>();
    return new Set(values.split(",").map((s) => s.trim()).filter(Boolean));
  }, [values]);

  const toggleChoice = useCallback((choice: string) => {
    const tokens = values.split(",").map((s) => s.trim()).filter(Boolean);
    const set = new Set(tokens);
    if (set.has(choice)) {
      set.delete(choice);
      onValuesChange([...set].join(", "));
    } else {
      onValuesChange([...tokens, choice].join(", "));
    }
  }, [values, onValuesChange]);

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
        <div className="pl-8 space-y-1">
          <div className="flex gap-1 items-center">
            <Input
              value={values}
              onChange={(e) => onValuesChange(e.target.value)}
              placeholder={selectedOption?.type === "int" || selectedOption?.type === "float" ? "10, 20, 30 or 10-50:5" : "value1, value2, value3"}
              className="h-8 text-2xs"
            />
            {values.trim() && (
              <Button variant="ghost" size="icon" className="h-8 w-8 shrink-0" onClick={onValidate} disabled={validation?.isLoading}>
                {validation?.isLoading ? <Loader2 size={12} className="animate-spin" /> : <CheckCircle2 size={12} />}
              </Button>
            )}
          </div>
          {expandedChoices.length > 0 && (
            <div>
              {expandedChoices.length > 15 && (
                <Input
                  value={choiceFilter}
                  onChange={(e) => setChoiceFilter(e.target.value)}
                  placeholder="Filter choices..."
                  className="h-6 text-3xs w-28 mb-1"
                />
              )}
              <div className="flex flex-wrap gap-1 max-h-20 overflow-y-auto">
                {filteredChoices.map((choice) => {
                  const isSelected = selectedSet.has(choice);
                  return (
                    <button
                      key={choice}
                      type="button"
                      onClick={() => toggleChoice(choice)}
                      className={cn(
                        "px-1.5 py-0.5 rounded text-3xs border transition-colors cursor-pointer",
                        isSelected
                          ? "bg-primary text-primary-foreground border-primary"
                          : "bg-muted text-muted-foreground border-border/50 hover:border-border hover:text-foreground"
                      )}
                    >
                      {choice}
                    </button>
                  );
                })}
              </div>
            </div>
          )}
          {validation?.response && !validation.response.ok && (
            <div className="text-2xs text-destructive">
              {validation.response.errors.map((e, i) => <div key={i}>{e}</div>)}
            </div>
          )}
          {validation?.response?.ok && (
            <div className="text-2xs text-muted-foreground">
              Resolved: {validation.response.resolved.map(String).join(", ")} ({validation.response.count} values)
            </div>
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
  const [xValidation, setXValidation] = useState<AxisValidation>({ isLoading: false });
  const [yValidation, setYValidation] = useState<AxisValidation>({ isLoading: false });
  const [zValidation, setZValidation] = useState<AxisValidation>({ isLoading: false });
  const [preview, setPreview] = useState<XyzPreviewResponse | null>(null);

  const { data: axisOptions } = useXyzAxisOptions();
  const submitJob = useSubmitJob();
  const trackJob = useJobQueueStore((s) => s.trackJob);
  const validateMutation = useXyzValidate();
  const previewMutation = useXyzPreview();

  const groups = useMemo(() => (axisOptions ? groupAxisOptions(axisOptions) : []), [axisOptions]);

  const allOptions = useMemo(() => axisOptions ?? [], [axisOptions]);
  const findOpt = useCallback((label: string) => allOptions.find((o) => o.label === label), [allOptions]);

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
    return parts.join(" \u00d7 ");
  }, [xType, yType, zType, showZ, xCount, yCount, zCount]);

  const canSubmit = (xType && xValues.trim()) || (yType && yValues.trim()) || (showZ && zType && zValues.trim());

  const handleValidate = useCallback((axisType: string, values: string, setter: (v: AxisValidation) => void) => {
    if (!axisType || !values.trim()) return;
    setter({ isLoading: true });
    validateMutation.mutate({ axis_type: axisType, values }, {
      onSuccess: (response) => setter({ response, isLoading: false }),
      onError: () => setter({ isLoading: false }),
    });
  }, [validateMutation]);

  const handlePreview = useCallback(() => {
    previewMutation.mutate({
      x_axis: xType && xValues.trim() ? { type: xType, values: xValues } : null,
      y_axis: yType && yValues.trim() ? { type: yType, values: yValues } : null,
      z_axis: showZ && zType && zValues.trim() ? { type: zType, values: zValues } : null,
    }, {
      onSuccess: (response) => setPreview(response),
    });
  }, [previewMutation, xType, xValues, yType, yValues, zType, zValues, showZ]);

  const handleSubmit = useCallback(async () => {
    if (!axisOptions) return;
    setIsSubmitting(true);
    try {
      const { payload, snapshot } = await buildRequest();
      // Extract generation params from the base payload, then overlay xyz-grid fields
      const { type: _type, script_name: _sn, script_args: _sa, priority: basePriority, ...baseParams } = payload as Record<string, unknown> & { type: string; script_name?: string; script_args?: unknown[]; priority?: number };
      const xyzPayload: XyzGridJobParams = {
        ...(baseParams as Omit<XyzGridJobParams, "type">),
        type: "xyz-grid",
        x_axis: xType && xValues.trim() ? { type: xType, values: xValues } : undefined,
        y_axis: yType && yValues.trim() ? { type: yType, values: yValues } : undefined,
        z_axis: showZ && zType && zValues.trim() ? { type: zType, values: zValues } : undefined,
        draw_legend: drawLegend,
        include_grid: true,
        include_subgrids: includeSubgrids,
        include_images: includeImages,
        include_time: includeTime,
        include_text: false,
        margin_size: 0,
        random_seeds: false,
      };
      const priority = basePriority ?? 0;

      const job = await submitJob.mutateAsync(xyzPayload);
      trackJob(job.id, "xyz-grid", snapshot, xyzPayload, priority);
      putJobPayload({ id: job.id, domain: "xyz-grid", request: xyzPayload, priority, snapshot: { controlUnits: snapshot.controlUnits }, createdAt: Date.now() });
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
          <AxisSection label="X" axisType={xType} onAxisTypeChange={setXType} values={xValues} onValuesChange={setXValues} groups={groups} allOptions={allOptions} validation={xValidation} onValidate={() => handleValidate(xType, xValues, setXValidation)} />
          <AxisSection label="Y" axisType={yType} onAxisTypeChange={setYType} values={yValues} onValuesChange={setYValues} groups={groups} allOptions={allOptions} validation={yValidation} onValidate={() => handleValidate(yType, yValues, setYValidation)} />

          {showZ ? (
            <>
              <AxisSection label="Z" axisType={zType} onAxisTypeChange={setZType} values={zValues} onValuesChange={setZValues} groups={groups} allOptions={allOptions} validation={zValidation} onValidate={() => handleValidate(zType, zValues, setZValidation)} />
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
              {preview && (
                <span className="text-muted-foreground ml-1">~{preview.total_steps} steps</span>
              )}
              {totalCells > 50 && (
                <span className="ml-auto flex items-center gap-1 text-amber-500">
                  <AlertTriangle size={12} /> Large grid
                </span>
              )}
              <Button variant="ghost" size="sm" className="ml-auto h-6 text-2xs" onClick={handlePreview} disabled={previewMutation.isPending}>
                {previewMutation.isPending ? <Loader2 size={10} className="animate-spin mr-1" /> : null}
                Preview
              </Button>
            </div>
          )}

          {preview && preview.errors.length > 0 && (
            <div className="text-2xs text-destructive px-3">
              {preview.errors.map((e, i) => <div key={i}>{e}</div>)}
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
