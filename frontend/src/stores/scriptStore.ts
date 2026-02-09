import { create } from "zustand";

interface ScriptState {
  selectedScript: string;
  scriptArgs: unknown[];
  alwaysOnOverrides: Record<string, unknown[]>;
  alwaysOnEnabled: Record<string, boolean>;

  setSelectedScript: (name: string) => void;
  setScriptArg: (index: number, value: unknown) => void;
  setAlwaysOnArg: (name: string, index: number, value: unknown) => void;
  setAlwaysOnEnabled: (name: string, enabled: boolean) => void;
  reset: () => void;
}

export const useScriptStore = create<ScriptState>()((set) => ({
  selectedScript: "",
  scriptArgs: [],
  alwaysOnOverrides: {},
  alwaysOnEnabled: {},

  setSelectedScript: (name) => set({ selectedScript: name, scriptArgs: [] }),

  setScriptArg: (index, value) =>
    set((state) => {
      const args = [...state.scriptArgs];
      while (args.length <= index) args.push(null);
      args[index] = value;
      return { scriptArgs: args };
    }),

  setAlwaysOnArg: (name, index, value) =>
    set((state) => {
      const overrides = { ...state.alwaysOnOverrides };
      const args = [...(overrides[name] ?? [])];
      while (args.length <= index) args.push(null);
      args[index] = value;
      overrides[name] = args;
      return { alwaysOnOverrides: overrides };
    }),

  setAlwaysOnEnabled: (name, enabled) =>
    set((state) => ({ alwaysOnEnabled: { ...state.alwaysOnEnabled, [name]: enabled } })),

  reset: () => set({ selectedScript: "", scriptArgs: [], alwaysOnOverrides: {}, alwaysOnEnabled: {} }),
}));
