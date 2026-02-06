import { create } from "zustand";
import type { IPAdapterUnit } from "@/api/types/adapter";

function defaultUnit(): IPAdapterUnit {
  return { adapter: "None", scale: 0.5, crop: false, start: 0, end: 1, images: [], masks: [] };
}

interface AdapterState {
  activeUnits: number;
  units: IPAdapterUnit[];
  unloadAdapter: boolean;

  setActiveUnits: (n: number) => void;
  setUnitParam: <K extends keyof IPAdapterUnit>(index: number, key: K, value: IPAdapterUnit[K]) => void;
  addUnitImage: (index: number, file: File) => void;
  removeUnitImage: (index: number, imageIdx: number) => void;
  addUnitMask: (index: number, file: File) => void;
  removeUnitMask: (index: number, maskIdx: number) => void;
  reset: () => void;
}

export const useAdapterStore = create<AdapterState>()((set) => ({
  activeUnits: 1,
  units: [defaultUnit(), defaultUnit(), defaultUnit(), defaultUnit()],
  unloadAdapter: false,

  setActiveUnits: (n) => set({ activeUnits: Math.max(1, Math.min(4, n)) }),

  setUnitParam: (index, key, value) =>
    set((state) => {
      const units = [...state.units];
      units[index] = { ...units[index], [key]: value };
      return { units };
    }),

  addUnitImage: (index, file) =>
    set((state) => {
      const units = [...state.units];
      units[index] = { ...units[index], images: [...units[index].images, file] };
      return { units };
    }),

  removeUnitImage: (index, imageIdx) =>
    set((state) => {
      const units = [...state.units];
      units[index] = { ...units[index], images: units[index].images.filter((_, i) => i !== imageIdx) };
      return { units };
    }),

  addUnitMask: (index, file) =>
    set((state) => {
      const units = [...state.units];
      units[index] = { ...units[index], masks: [...units[index].masks, file] };
      return { units };
    }),

  removeUnitMask: (index, maskIdx) =>
    set((state) => {
      const units = [...state.units];
      units[index] = { ...units[index], masks: units[index].masks.filter((_, i) => i !== maskIdx) };
      return { units };
    }),

  reset: () => set({ activeUnits: 1, units: [defaultUnit(), defaultUnit(), defaultUnit(), defaultUnit()], unloadAdapter: false }),
}));
