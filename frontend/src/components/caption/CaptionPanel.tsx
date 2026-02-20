import { useCallback } from "react";
import { Play, Loader2 } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useCaptionStore } from "@/stores/captionStore";
import { useCaptionSettingsStore } from "@/stores/captionSettingsStore";
import { useOpenClipCaption, useTaggerCaption, useVqaCaption } from "@/api/hooks/useCaption";
import { uploadFile } from "@/lib/upload";
import { CUSTOM_PROMPT_TASKS } from "@/lib/captionModels";
import { VlmSettings } from "./methods/VlmSettings";
import { OpenClipSettings } from "./methods/OpenClipSettings";
import { TaggerSettings } from "./methods/TaggerSettings";
import type { CaptionMethod } from "@/api/types/caption";

export function CaptionPanel() {
  const image = useCaptionStore((s) => s.image);
  const isProcessing = useCaptionStore((s) => s.isProcessing);
  const method = useCaptionStore((s) => s.method);
  const setResult = useCaptionStore((s) => s.setResult);
  const setProcessing = useCaptionStore((s) => s.setProcessing);
  const setMethod = useCaptionStore((s) => s.setMethod);

  const openclipMut = useOpenClipCaption();
  const taggerMut = useTaggerCaption();
  const vqaMut = useVqaCaption();

  const handleCaption = useCallback(async () => {
    if (!image || isProcessing) return;
    setProcessing(true);
    setResult(null);
    try {
      const ref = await uploadFile(image);
      const settings = useCaptionSettingsStore.getState();

      if (method === "vlm") {
        const s = settings.vlm;
        const res = await vqaMut.mutateAsync({
          image: ref,
          model: s.model,
          question: s.task,
          prompt: CUSTOM_PROMPT_TASKS.includes(s.task) ? s.customPrompt : undefined,
          system: s.system,
          include_annotated: s.includeAnnotated,
          max_tokens: s.maxTokens,
          temperature: s.temperature,
          top_k: s.topK,
          top_p: s.topP,
          num_beams: s.numBeams,
          do_sample: s.doSample,
          thinking_mode: s.thinkingMode,
          prefill: s.prefill || undefined,
          keep_thinking: s.keepThinking,
          keep_prefill: s.keepPrefill,
        });
        setResult({ ...res, type: "vqa" });
      } else if (method === "openclip") {
        const s = settings.openclip;
        const res = await openclipMut.mutateAsync({
          image: ref,
          clip_model: s.clipModel,
          blip_model: s.blipModel,
          mode: s.mode,
          analyze: s.analyze,
          max_length: s.maxLength,
          chunk_size: s.chunkSize,
          min_flavors: s.minFlavors,
          max_flavors: s.maxFlavors,
          flavor_count: s.flavorCount,
          num_beams: s.numBeams,
        });
        setResult({ ...res, type: "openclip" });
      } else {
        const s = settings.tagger;
        const res = await taggerMut.mutateAsync({
          image: ref,
          model: s.model,
          threshold: s.threshold,
          character_threshold: s.characterThreshold,
          max_tags: s.maxTags,
          include_rating: s.includeRating,
          sort_alpha: s.sortAlpha,
          use_spaces: s.useSpaces,
          escape_brackets: s.escapeBrackets,
          exclude_tags: s.excludeTags,
          show_scores: s.showScores,
        });
        setResult({ ...res, type: "tagger" });
      }
    } catch (err) {
      toast.error("Captioning failed", { description: err instanceof Error ? err.message : String(err) });
    } finally {
      setProcessing(false);
    }
  }, [image, isProcessing, method, vqaMut, openclipMut, taggerMut, setProcessing, setResult]);

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

      {/* Method tabs + settings */}
      <ScrollArea className="flex-1">
        <div className="p-3">
          <Tabs value={method} onValueChange={(v) => setMethod(v as CaptionMethod)}>
            <TabsList className="w-full">
              <TabsTrigger value="vlm" className="text-xs flex-1">VLM</TabsTrigger>
              <TabsTrigger value="openclip" className="text-xs flex-1">OpenCLiP</TabsTrigger>
              <TabsTrigger value="tagger" className="text-xs flex-1">Tagger</TabsTrigger>
            </TabsList>

            <TabsContent value="vlm" className="mt-3">
              <VlmSettings />
            </TabsContent>

            <TabsContent value="openclip" className="mt-3">
              <OpenClipSettings />
            </TabsContent>

            <TabsContent value="tagger" className="mt-3">
              <TaggerSettings />
            </TabsContent>
          </Tabs>
        </div>
      </ScrollArea>
    </div>
  );
}
