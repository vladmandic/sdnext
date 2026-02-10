import { useCallback, useRef } from "react";
import { Play, Loader2, Copy, ArrowRight } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useCaptionStore } from "@/stores/captionStore";
import { useGenerationStore } from "@/stores/generationStore";
import { useInterrogate, useVqa, setCaptionOptions } from "@/api/hooks/useCaption";
import { fileToBase64 } from "@/lib/image";
import { VlmSettings, type VlmSettingsValues } from "./methods/VlmSettings";
import { OpenClipSettings, type OpenClipSettingsValues } from "./methods/OpenClipSettings";
import { TaggerSettings, type TaggerSettingsValues } from "./methods/TaggerSettings";
import type { CaptionMethod } from "@/api/types/caption";

export function CaptionPanel() {
  const image = useCaptionStore((s) => s.image);
  const result = useCaptionStore((s) => s.result);
  const isProcessing = useCaptionStore((s) => s.isProcessing);
  const method = useCaptionStore((s) => s.method);
  const setResult = useCaptionStore((s) => s.setResult);
  const setProcessing = useCaptionStore((s) => s.setProcessing);
  const setMethod = useCaptionStore((s) => s.setMethod);

  const interrogate = useInterrogate();
  const vqa = useVqa();

  // Refs to hold current settings from method components
  const vlmRef = useRef<VlmSettingsValues>({ model: "Alibaba Qwen 2.5 VL 3B", question: "<DETAILED_CAPTION>", system: "", options: {} });
  const clipRef = useRef<OpenClipSettingsValues>({ model: "", blipModel: "blip-base", mode: "fast", analyze: false, options: {} });
  const taggerRef = useRef<TaggerSettingsValues>({ model: "wd-eva02-large-tagger-v3", options: {} });

  const handleCaption = useCallback(async () => {
    if (!image || isProcessing) return;
    setProcessing(true);
    setResult(null);
    try {
      const base64 = await fileToBase64(image);
      if (method === "vlm") {
        const { model, question, system, options } = vlmRef.current;
        await setCaptionOptions(options);
        const res = await vqa.mutateAsync({ image: base64, model, question, system });
        setResult({ ...res, type: "vqa" });
      } else if (method === "openclip") {
        const { model, blipModel, mode, analyze, options } = clipRef.current;
        await setCaptionOptions(options);
        const res = await interrogate.mutateAsync({ image: base64, model, blip_model: blipModel, mode: mode as "fast" | "classic" | "best" | "negative", analyze });
        setResult({ ...res, type: "interrogate" });
      } else {
        const { model, options } = taggerRef.current;
        await setCaptionOptions(options);
        // DeepBooru uses the interrogate endpoint; WaifuDiffusion models use options + deepdanbooru path
        const apiModel = model === "DeepBooru" ? "deepdanbooru" : "deepdanbooru";
        const res = await interrogate.mutateAsync({ image: base64, model: apiModel });
        setResult({ ...res, type: "interrogate" });
      }
    } catch (err) {
      toast.error("Captioning failed", { description: err instanceof Error ? err.message : String(err) });
    } finally {
      setProcessing(false);
    }
  }, [image, isProcessing, method, vqa, interrogate, setProcessing, setResult]);

  const resultText = getResultText(result);

  const handleCopy = useCallback(() => {
    if (resultText) {
      navigator.clipboard.writeText(resultText);
      toast.success("Copied to clipboard");
    }
  }, [resultText]);

  const handleToPrompt = useCallback(() => {
    if (!resultText) return;
    const current = useGenerationStore.getState().prompt;
    useGenerationStore.getState().setParam("prompt", current ? `${current}, ${resultText}` : resultText);
    toast.success("Appended to prompt");
  }, [resultText]);

  const handleToNegative = useCallback(() => {
    if (!resultText) return;
    const current = useGenerationStore.getState().negativePrompt;
    useGenerationStore.getState().setParam("negativePrompt", current ? `${current}, ${resultText}` : resultText);
    toast.success("Appended to negative prompt");
  }, [resultText]);

  return (
    <div className="flex flex-col h-full">
      {/* Caption button */}
      <div className="px-3 py-2 border-b border-border">
        <Button
          onClick={handleCaption}
          disabled={!image || isProcessing}
          size="sm"
          className="w-full"
        >
          {isProcessing ? (
            <>
              <Loader2 size={14} className="animate-spin" />
              Captioning...
            </>
          ) : (
            <>
              <Play size={14} />
              Caption
            </>
          )}
        </Button>
      </div>

      {/* Method tabs + settings + image upload */}
      <ScrollArea className="flex-1">
        <div className="p-3">
          <Tabs value={method} onValueChange={(v) => setMethod(v as CaptionMethod)}>
            <TabsList className="w-full">
              <TabsTrigger value="vlm" className="text-xs flex-1">VLM</TabsTrigger>
              <TabsTrigger value="openclip" className="text-xs flex-1">OpenCLiP</TabsTrigger>
              <TabsTrigger value="tagger" className="text-xs flex-1">Tagger</TabsTrigger>
            </TabsList>

            <TabsContent value="vlm" className="mt-3">
              <VlmSettings onChange={(v) => { vlmRef.current = v; }} />
            </TabsContent>

            <TabsContent value="openclip" className="mt-3">
              <OpenClipSettings onChange={(v) => { clipRef.current = v; }} />
            </TabsContent>

            <TabsContent value="tagger" className="mt-3">
              <TaggerSettings onChange={(v) => { taggerRef.current = v; }} />
            </TabsContent>
          </Tabs>
        </div>
      </ScrollArea>

      {/* Results area */}
      {result && (
        <div className="border-t border-border p-3 flex flex-col gap-2 max-h-[300px] overflow-y-auto">
          <Label className="text-xs text-muted-foreground">Result</Label>

          {result.type === "interrogate" && result.caption && (
            <Textarea value={result.caption} readOnly className="min-h-12 text-xs" rows={3} />
          )}
          {result.type === "vqa" && result.answer && (
            <Textarea value={result.answer} readOnly className="min-h-12 text-xs" rows={3} />
          )}

          {/* Analyze breakdown for OpenCLiP */}
          {result.type === "interrogate" && (result.medium || result.artist || result.movement || result.trending || result.flavor) && (
            <div className="flex flex-col gap-1 text-[10px]">
              {result.medium && <AnalyzeField label="Medium" value={result.medium} />}
              {result.artist && <AnalyzeField label="Artist" value={result.artist} />}
              {result.movement && <AnalyzeField label="Movement" value={result.movement} />}
              {result.trending && <AnalyzeField label="Trending" value={result.trending} />}
              {result.flavor && <AnalyzeField label="Flavor" value={result.flavor} />}
            </div>
          )}

          {/* Action buttons */}
          {resultText && (
            <div className="flex items-center gap-1.5">
              <Button variant="secondary" size="sm" className="text-xs h-7 gap-1" onClick={handleCopy}>
                <Copy size={10} />
                Copy
              </Button>
              <Button variant="secondary" size="sm" className="text-xs h-7 gap-1" onClick={handleToPrompt}>
                <ArrowRight size={10} />
                Prompt
              </Button>
              <Button variant="secondary" size="sm" className="text-xs h-7 gap-1" onClick={handleToNegative}>
                <ArrowRight size={10} />
                Negative
              </Button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function AnalyzeField({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <span className="text-muted-foreground">{label}: </span>
      <span className="text-foreground">{value}</span>
    </div>
  );
}

function getResultText(result: ReturnType<typeof useCaptionStore.getState>["result"]): string {
  if (!result) return "";
  if (result.type === "vqa") return result.answer ?? "";
  return result.caption ?? "";
}
