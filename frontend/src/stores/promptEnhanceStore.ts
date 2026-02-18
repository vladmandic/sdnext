import { create } from "zustand";
import { persist } from "zustand/middleware";

interface PromptEnhanceState {
  model: string;
  systemPrompt: string;
  prefix: string;
  suffix: string;
  nsfw: boolean;
  seed: number;
  doSample: boolean;
  maxTokens: number;
  temperature: number;
  repetitionPenalty: number;
  topK: number;
  topP: number;
  thinking: boolean;
  keepThinking: boolean;
  useVision: boolean;
  prefill: string;
  keepPrefill: boolean;

  setModel: (model: string) => void;
  setSystemPrompt: (v: string) => void;
  setPrefix: (v: string) => void;
  setSuffix: (v: string) => void;
  setNsfw: (v: boolean) => void;
  setSeed: (v: number) => void;
  setDoSample: (v: boolean) => void;
  setMaxTokens: (v: number) => void;
  setTemperature: (v: number) => void;
  setRepetitionPenalty: (v: number) => void;
  setTopK: (v: number) => void;
  setTopP: (v: number) => void;
  setThinking: (v: boolean) => void;
  setKeepThinking: (v: boolean) => void;
  setUseVision: (v: boolean) => void;
  setPrefill: (v: string) => void;
  setKeepPrefill: (v: boolean) => void;
}

export const usePromptEnhanceStore = create<PromptEnhanceState>()(
  persist(
    (set) => ({
      model: "",
      systemPrompt: "",
      prefix: "",
      suffix: "",
      nsfw: true,
      seed: -1,
      doSample: true,
      maxTokens: 512,
      temperature: 0.8,
      repetitionPenalty: 1.2,
      topK: 0,
      topP: 0,
      thinking: false,
      keepThinking: false,
      useVision: true,
      prefill: "",
      keepPrefill: false,

      setModel: (model) => set({ model }),
      setSystemPrompt: (v) => set({ systemPrompt: v }),
      setPrefix: (v) => set({ prefix: v }),
      setSuffix: (v) => set({ suffix: v }),
      setNsfw: (v) => set({ nsfw: v }),
      setSeed: (v) => set({ seed: v }),
      setDoSample: (v) => set({ doSample: v }),
      setMaxTokens: (v) => set({ maxTokens: v }),
      setTemperature: (v) => set({ temperature: v }),
      setRepetitionPenalty: (v) => set({ repetitionPenalty: v }),
      setTopK: (v) => set({ topK: v }),
      setTopP: (v) => set({ topP: v }),
      setThinking: (v) => set({ thinking: v }),
      setKeepThinking: (v) => set({ keepThinking: v }),
      setUseVision: (v) => set({ useVision: v }),
      setPrefill: (v) => set({ prefill: v }),
      setKeepPrefill: (v) => set({ keepPrefill: v }),
    }),
    { name: "sdnext-enhance" },
  ),
);
