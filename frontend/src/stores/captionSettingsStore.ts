import { create } from "zustand";
import { persist } from "zustand/middleware";
import { VLM_DEFAULT, VLM_SYSTEM_DEFAULT, TAGGER_DEFAULT } from "@/lib/captionModels";

interface VlmSettings {
  model: string;
  task: string;
  customPrompt: string;
  system: string;
  maxTokens: number;
  temperature: number;
  topK: number;
  topP: number;
  numBeams: number;
  doSample: boolean;
  thinkingMode: boolean;
  prefill: string;
  keepThinking: boolean;
  keepPrefill: boolean;
}

interface OpenClipSettings {
  clipModel: string;
  blipModel: string;
  mode: string;
  analyze: boolean;
  maxLength: number;
  chunkSize: number;
  minFlavors: number;
  maxFlavors: number;
  flavorCount: number;
  numBeams: number;
}

interface TaggerSettings {
  model: string;
  threshold: number;
  characterThreshold: number;
  maxTags: number;
  includeRating: boolean;
  sortAlpha: boolean;
  useSpaces: boolean;
  escapeBrackets: boolean;
  excludeTags: string;
  showScores: boolean;
}

interface CaptionSettingsState {
  vlm: VlmSettings;
  openclip: OpenClipSettings;
  tagger: TaggerSettings;
  setVlm: (patch: Partial<VlmSettings>) => void;
  setOpenClip: (patch: Partial<OpenClipSettings>) => void;
  setTagger: (patch: Partial<TaggerSettings>) => void;
}

export const useCaptionSettingsStore = create<CaptionSettingsState>()(
  persist(
    (set) => ({
      vlm: {
        model: VLM_DEFAULT,
        task: "Normal Caption",
        customPrompt: "",
        system: VLM_SYSTEM_DEFAULT,
        maxTokens: 512,
        temperature: 0.8,
        topK: 0,
        topP: 0,
        numBeams: 1,
        doSample: true,
        thinkingMode: false,
        prefill: "",
        keepThinking: false,
        keepPrefill: false,
      },
      openclip: {
        clipModel: "ViT-L-14/openai",
        blipModel: "blip-base",
        mode: "fast",
        analyze: false,
        maxLength: 74,
        chunkSize: 1024,
        minFlavors: 2,
        maxFlavors: 16,
        flavorCount: 1024,
        numBeams: 1,
      },
      tagger: {
        model: TAGGER_DEFAULT,
        threshold: 0.5,
        characterThreshold: 0.85,
        maxTags: 74,
        includeRating: false,
        sortAlpha: false,
        useSpaces: false,
        escapeBrackets: true,
        excludeTags: "",
        showScores: false,
      },
      setVlm: (patch) => set((s) => ({ vlm: { ...s.vlm, ...patch } })),
      setOpenClip: (patch) => set((s) => ({ openclip: { ...s.openclip, ...patch } })),
      setTagger: (patch) => set((s) => ({ tagger: { ...s.tagger, ...patch } })),
    }),
    {
      name: "sdnext-caption-settings",
      merge: (persisted, current) => {
        const p = persisted as Partial<CaptionSettingsState> | undefined;
        return {
          ...current,
          vlm: { ...current.vlm, ...p?.vlm },
          openclip: { ...current.openclip, ...p?.openclip },
          tagger: { ...current.tagger, ...p?.tagger },
        };
      },
    },
  ),
);
