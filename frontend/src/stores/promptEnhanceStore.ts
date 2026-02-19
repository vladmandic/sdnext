import { create } from "zustand";
import { persist } from "zustand/middleware";

export interface EnhanceHistoryEntry {
  id: string;
  prompt: string;
  originalPrompt: string;
  seed: number;
  timestamp: number;
}

interface PendingResult {
  prompt: string;
  seed: number;
  originalPrompt: string;
}

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

  pinned: boolean;
  pendingResult: PendingResult | null;
  history: EnhanceHistoryEntry[];
  historyLimit: number;

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

  setPinned: (v: boolean) => void;
  setPendingResult: (v: PendingResult | null) => void;
  addToHistory: (entry: Omit<EnhanceHistoryEntry, "id" | "timestamp">) => void;
  clearHistory: () => void;
  setHistoryLimit: (limit: number) => void;
}

export const usePromptEnhanceStore = create<PromptEnhanceState>()(
  persist(
    (set, get) => ({
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

      pinned: false,
      pendingResult: null,
      history: [],
      historyLimit: 50,

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

      setPinned: (v) => set({ pinned: v }),
      setPendingResult: (v) => set({ pendingResult: v }),
      addToHistory: (entry) => {
        const { history, historyLimit } = get();
        const newEntry: EnhanceHistoryEntry = {
          ...entry,
          id: crypto.randomUUID(),
          timestamp: Date.now(),
        };
        set({ history: [newEntry, ...history].slice(0, historyLimit) });
      },
      clearHistory: () => set({ history: [] }),
      setHistoryLimit: (limit) => set({ historyLimit: limit }),
    }),
    {
      name: "sdnext-enhance",
      partialize: (state) => ({
        model: state.model,
        systemPrompt: state.systemPrompt,
        prefix: state.prefix,
        suffix: state.suffix,
        nsfw: state.nsfw,
        seed: state.seed,
        doSample: state.doSample,
        maxTokens: state.maxTokens,
        temperature: state.temperature,
        repetitionPenalty: state.repetitionPenalty,
        topK: state.topK,
        topP: state.topP,
        thinking: state.thinking,
        keepThinking: state.keepThinking,
        useVision: state.useVision,
        prefill: state.prefill,
        keepPrefill: state.keepPrefill,
        pinned: state.pinned,
        history: state.history,
        historyLimit: state.historyLimit,
        // pendingResult excluded — transient
      }),
    },
  ),
);
