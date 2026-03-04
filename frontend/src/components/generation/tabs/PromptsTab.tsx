import { useMemo, useState, useCallback } from "react";
import { useGenerationStore } from "@/stores/generationStore";
import { useUiStore } from "@/stores/uiStore";
import { useImg2ImgStore } from "@/stores/img2imgStore";
import { useIsImg2Img } from "@/hooks/useIsImg2Img";
import { useShallow } from "zustand/react/shallow";
import { usePromptStyles } from "@/api/hooks/useNetworks";
import { useOptionsSubset } from "@/api/hooks/useSettings";
import { useUpscalerGroups } from "@/api/hooks/useModels";
import { Link2Off, ArrowLeftRight, ChevronDown, X } from "lucide-react";
import { cn } from "@/lib/utils";
import { resolveGenerationSize, formatMegapixels } from "@/lib/sizeCompute";
import type { SizeMode } from "@/lib/sizeCompute";
import { PromptEditor } from "../PromptEditor";
import { ParamSlider } from "../ParamSlider";
import { ParamSection } from "../ParamSection";
import { ParamGrid } from "../ParamRow";
import { NumberInput } from "@/components/ui/number-input";
import { ParamLabel } from "../ParamLabel";
import { Button } from "@/components/ui/button";
import { Combobox } from "@/components/ui/combobox";
import { Popover, PopoverTrigger, PopoverContent } from "@/components/ui/popover";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";

interface AspectPreset { label: string; w: number; h: number }

function parseAspectRatios(raw: string): AspectPreset[] {
  return raw.split(",").map((s) => s.trim()).filter(Boolean).map((s) => {
    const [w, h] = s.split(":").map(Number);
    return (w > 0 && h > 0) ? { label: s, w, h } : null;
  }).filter((p): p is AspectPreset => p !== null);
}

export function PromptsTab() {
  const state = useGenerationStore(useShallow((s) => ({
    width: s.width,
    height: s.height,
    batchCount: s.batchCount,
    batchSize: s.batchSize,
  })));
  const setParam = useGenerationStore((s) => s.setParam);
  const { data: styles } = usePromptStyles();
  const isImg2Img = useIsImg2Img();
  const autoFitFrame = useUiStore((s) => s.autoFitFrame);
  const setAutoFitFrame = useUiStore((s) => s.setAutoFitFrame);
  const sizeMode = useImg2ImgStore((s) => s.sizeMode);
  const setSizeMode = useImg2ImgStore((s) => s.setSizeMode);
  const scaleFactor = useImg2ImgStore((s) => s.scaleFactor);
  const setScaleFactor = useImg2ImgStore((s) => s.setScaleFactor);
  const megapixelTarget = useImg2ImgStore((s) => s.megapixelTarget);
  const setMegapixelTarget = useImg2ImgStore((s) => s.setMegapixelTarget);
  const resizeMethod = useImg2ImgStore((s) => s.resizeMethod);
  const setResizeMethod = useImg2ImgStore((s) => s.setResizeMethod);
  const upscalerGroups = useUpscalerGroups({ excludeLatent: true });
  const { data: aspectOpts } = useOptionsSubset(["aspect_ratios"]);
  const aspectPresets = useMemo(() => parseAspectRatios(typeof aspectOpts?.aspect_ratios === "string" ? aspectOpts.aspect_ratios : "1:1, 4:3, 3:2, 16:9, 16:10, 21:9, 2:3, 3:4, 9:16, 10:16, 9:21"), [aspectOpts]);
  const [activePreset, setActivePreset] = useState<string | null>(null);
  const aspectLocked = activePreset !== null;
  const [aspectOpen, setAspectOpen] = useState(false);
  const [selectedStyles, setSelectedStyles] = useState<string[]>([]);

  const showSizeModes = isImg2Img && autoFitFrame;
  const effectiveSizeMode: SizeMode = showSizeModes ? sizeMode : "fixed";
  const isFixed = effectiveSizeMode === "fixed";

  const genSize = useMemo(
    () => resolveGenerationSize(effectiveSizeMode, state.width, state.height, scaleFactor, megapixelTarget),
    [effectiveSizeMode, state.width, state.height, scaleFactor, megapixelTarget],
  );

  const lockedRatio = useMemo(() => {
    if (!activePreset) return null;
    const [w, h] = activePreset.split(":").map(Number);
    return (w > 0 && h > 0) ? w / h : null;
  }, [activePreset]);

  const setWidth = useCallback((w: number) => {
    const rounded = Math.round(w / 8) * 8;
    setParam("width", rounded);
    if (lockedRatio) setParam("height", Math.round(rounded / lockedRatio / 8) * 8);
  }, [setParam, lockedRatio]);

  const setHeight = useCallback((h: number) => {
    const rounded = Math.round(h / 8) * 8;
    setParam("height", rounded);
    if (lockedRatio) setParam("width", Math.round(rounded * lockedRatio / 8) * 8);
  }, [setParam, lockedRatio]);

  const swapDimensions = useCallback(() => {
    const w = state.width;
    setParam("width", state.height);
    setParam("height", w);
    if (activePreset) {
      const parts = activePreset.split(":");
      if (parts.length === 2) setActivePreset(`${parts[1]}:${parts[0]}`);
    }
  }, [setParam, state.width, state.height, activePreset]);

  const selectPreset = useCallback((preset: AspectPreset | null) => {
    if (!preset) { setActivePreset(null); setAspectOpen(false); return; }
    setActivePreset(preset.label);
    const pixels = state.width * state.height;
    const ratio = preset.w / preset.h;
    const newW = Math.round(Math.sqrt(pixels * ratio) / 8) * 8;
    const newH = Math.round(newW / ratio / 8) * 8;
    setParam("width", newW);
    setParam("height", newH);
    setAspectOpen(false);
  }, [state.width, state.height, setParam]);

  const set = useMemo(() => ({
    batchCount: (v: number) => setParam("batchCount", v),
    batchSize: (v: number) => setParam("batchSize", v),
  }), [setParam]);

  function addStyle(name: string) {
    if (!selectedStyles.includes(name)) {
      setSelectedStyles([...selectedStyles, name]);
    }
  }

  function removeStyle(name: string) {
    setSelectedStyles(selectedStyles.filter((s) => s !== name));
  }

  return (
    <div className="flex flex-col gap-3 text-sm">
      <PromptEditor />

      {styles && styles.length > 0 && (
        <ParamSection title="Styles" defaultOpen={false}>
          <div className="flex flex-wrap gap-1 mb-1">
            {selectedStyles.map((name) => (
              <span key={name} className="inline-flex items-center gap-0.5 px-1.5 py-0.5 text-3xs bg-muted rounded">
                {name}
                <button onClick={() => removeStyle(name)} className="text-muted-foreground hover:text-foreground">
                  <X size={10} />
                </button>
              </span>
            ))}
          </div>
          <Combobox
            value=""
            onValueChange={addStyle}
            options={styles.filter((s) => !selectedStyles.includes(s.name)).map((s) => s.name)}
            placeholder="Add style..."
            className="h-6 text-2xs"
          />
        </ParamSection>
      )}

      <ParamSection
        title="Size"
        action={isImg2Img ? (
          <Button
            variant={autoFitFrame ? "default" : "outline"}
            size="sm"
            onClick={() => setAutoFitFrame(!autoFitFrame)}
            className="h-5 px-1.5 text-3xs rounded"
            title={autoFitFrame
              ? "Auto: dropping the first image onto an empty canvas resizes the frame to match that image's dimensions"
              : "Manual: frame stays at the width and height you set, regardless of image size"}
          >
            Auto
          </Button>
        ) : undefined}
      >
        {/* Size mode pill selector (img2img + auto-fit only) */}
        {showSizeModes && (
          <Tabs value={sizeMode} onValueChange={(v) => setSizeMode(v as SizeMode)}>
            <TabsList className="h-7 w-full">
              <TabsTrigger value="fixed" className="text-2xs h-5 px-2">Fixed</TabsTrigger>
              <TabsTrigger value="scale" className="text-2xs h-5 px-2">Scale</TabsTrigger>
              <TabsTrigger value="megapixel" className="text-2xs h-5 px-2">Megapixel</TabsTrigger>
            </TabsList>
          </Tabs>
        )}

        {/* Width / Height row */}
        <div data-param="width" className="flex items-center gap-2">
          <ParamLabel className="text-2xs text-muted-foreground shrink-0">Width</ParamLabel>
          <NumberInput
            value={isFixed ? state.width : genSize.width}
            onChange={setWidth}
            step={8} min={64} max={4096} fallback={512}
            disabled={!isFixed}
            className="flex-1 min-w-12 h-6 text-2xs text-center px-1 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
          />
          <Popover open={aspectOpen} onOpenChange={setAspectOpen}>
            <PopoverTrigger asChild>
              <button
                type="button"
                disabled={!isFixed}
                className={cn(
                  "inline-flex items-center justify-center gap-0 h-6 w-9 shrink-0 rounded-md transition-colors",
                  "hover:bg-accent hover:text-accent-foreground",
                  "disabled:pointer-events-none disabled:opacity-50",
                  aspectLocked ? "text-primary px-1" : "text-muted-foreground px-1",
                )}
                title={aspectLocked ? `Aspect ratio locked to ${activePreset}` : "Select aspect ratio preset"}
              >
                {aspectLocked ? <span className="text-3xs font-medium leading-none">{activePreset}</span> : <Link2Off size={12} />}
                <ChevronDown size={8} className="ml-0.5 opacity-60" />
              </button>
            </PopoverTrigger>
            <PopoverContent className="w-32 p-1" align="center" sideOffset={6}>
              <button
                type="button"
                onClick={() => selectPreset(null)}
                className={cn(
                  "w-full text-left text-2xs px-2 py-1 rounded-sm transition-colors",
                  "hover:bg-accent hover:text-accent-foreground",
                  !aspectLocked && "text-primary font-medium",
                )}
              >
                Custom
              </button>
              {aspectPresets.map((p) => (
                <button
                  key={p.label}
                  type="button"
                  onClick={() => selectPreset(p)}
                  className={cn(
                    "w-full text-left text-2xs px-2 py-1 rounded-sm transition-colors",
                    "hover:bg-accent hover:text-accent-foreground",
                    activePreset === p.label && "text-primary font-medium",
                  )}
                >
                  {p.label}
                </button>
              ))}
            </PopoverContent>
          </Popover>
          <Button
            variant="ghost" size="icon-xs"
            onClick={swapDimensions}
            className="text-muted-foreground"
            title="Swap width and height — switch between landscape and portrait"
            disabled={!isFixed}
          >
            <ArrowLeftRight size={12} />
          </Button>
          <NumberInput
            value={isFixed ? state.height : genSize.height}
            onChange={setHeight}
            step={8} min={64} max={4096} fallback={512}
            disabled={!isFixed}
            className="flex-1 min-w-12 h-6 text-2xs text-center px-1 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
          />
          <ParamLabel className="text-2xs text-muted-foreground shrink-0">Height</ParamLabel>
        </div>

        {/* Scale slider */}
        {effectiveSizeMode === "scale" && (
          <ParamSlider label="Scale" value={scaleFactor} onChange={setScaleFactor} min={0.25} max={2} step={0.05} />
        )}

        {/* Megapixel slider */}
        {effectiveSizeMode === "megapixel" && (
          <ParamSlider label="Target" value={megapixelTarget} onChange={setMegapixelTarget} min={0.25} max={4} step={0.05} />
        )}

        {/* Resize method (shown when scale/megapixel active) */}
        {!isFixed && (
          <div className="flex items-center gap-2">
            <ParamLabel className="text-2xs text-muted-foreground w-16 flex-shrink-0">Resize</ParamLabel>
            <Combobox
              value={resizeMethod}
              onValueChange={setResizeMethod}
              groups={upscalerGroups}
              className="h-6 text-2xs flex-1"
            />
          </div>
        )}

        {/* Info line: frame size → generation size */}
        {!isFixed && (
          <div className="text-3xs text-muted-foreground text-center">
            {state.width}&times;{state.height} &rarr; {genSize.width}&times;{genSize.height}{" "}
            <span className="opacity-70">({formatMegapixels(genSize.width, genSize.height)})</span>
          </div>
        )}
      </ParamSection>

      <ParamSection title="Batch">
        <ParamGrid>
          <ParamSlider label="Count" value={state.batchCount} onChange={set.batchCount} min={1} max={100} />
          <ParamSlider label="Size" value={state.batchSize} onChange={set.batchSize} min={1} max={16} />
        </ParamGrid>
      </ParamSection>
    </div>
  );
}
