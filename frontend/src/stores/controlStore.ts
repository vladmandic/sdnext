import { create } from "zustand";
import type { ControlUnit, ControlUnitType } from "@/api/types/control";

function defaultUnit(): ControlUnit {
  return { enabled: true, unitType: "controlnet", processor: "None", model: "None", strength: 1.0, start: 0, end: 1, image: null };
}

interface ControlState {
  activeType: ControlUnitType;
  guessMode: boolean;
  units: ControlUnit[];

  addUnit: () => void;
  removeUnit: (index: number) => void;
  setUnitParam: <K extends keyof ControlUnit>(index: number, key: K, value: ControlUnit[K]) => void;
  setUnitImage: (index: number, file: File | null) => void;
  setActiveType: (type: ControlUnitType) => void;
  reset: () => void;
}

export const useControlStore = create<ControlState>()((set) => ({
  activeType: "controlnet",
  guessMode: false,
  units: [defaultUnit()],

  addUnit: () =>
    set((state) => ({ units: [...state.units, defaultUnit()] })),

  removeUnit: (index) =>
    set((state) => ({
      units: state.units.length > 1 ? state.units.filter((_, i) => i !== index) : state.units,
    })),

  setUnitParam: (index, key, value) =>
    set((state) => {
      const units = [...state.units];
      units[index] = { ...units[index], [key]: value };
      return { units };
    }),

  setUnitImage: (index, file) =>
    set((state) => {
      const units = [...state.units];
      units[index] = { ...units[index], image: file };
      return { units };
    }),

  setActiveType: (type) => set({ activeType: type }),

  reset: () => set({ activeType: "controlnet", guessMode: false, units: [defaultUnit()] }),
}));
