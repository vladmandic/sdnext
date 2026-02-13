import { create } from "zustand";
import type { ControlUnit, ControlUnitType, ControlUnitSnapshot } from "@/api/types/control";
import { fileToBase64, base64ToFile } from "@/lib/image";

function defaultUnit(unitType: ControlUnitType = "asset"): ControlUnit {
  return {
    enabled: false,
    unitType,
    processor: "None",
    model: "None",
    mode: "default",
    strength: 1.0,
    start: 0,
    end: 1,
    useSeparateImage: false,
    image: null,
    processedImage: null,
    guess: false,
    factor: 1.0,
    attention: "Attention",
    fidelity: 0.5,
    queryWeight: 1.0,
    adainWeight: 1.0,
    adapter: "None",
    scale: 0.5,
    crop: false,
    images: [],
    masks: [],
    fitMode: "contain",
  };
}

interface ControlState {
  units: ControlUnit[];

  addUnit: () => void;
  removeUnit: (index: number) => void;
  setUnitCount: (count: number) => void;
  setUnitParam: <K extends keyof ControlUnit>(index: number, key: K, value: ControlUnit[K]) => void;
  setUnitImage: (index: number, file: File | null) => void;
  setUnitType: (index: number, unitType: ControlUnitType) => void;
  addUnitImage: (index: number, file: File) => void;
  removeUnitImage: (index: number, imageIdx: number) => void;
  toggleSeparateImage: (index: number) => void;
  addUnitMask: (index: number, file: File) => void;
  removeUnitMask: (index: number, maskIdx: number) => void;
  restoreUnits: (snapshots: ControlUnitSnapshot[]) => void;
  reset: () => void;
}

export const useControlStore = create<ControlState>()((set) => ({
  units: [defaultUnit()],

  addUnit: () =>
    set((state) => {
      if (state.units.length >= 10) return state;
      const last = state.units.at(-1);
      const newUnit = {
        ...defaultUnit(last?.unitType),
        enabled: true,
        useSeparateImage: last?.useSeparateImage ?? false,
      };
      return { units: [...state.units, newUnit] };
    }),

  removeUnit: (index) =>
    set((state) => ({
      units: state.units.length > 1 ? state.units.filter((_, i) => i !== index) : state.units,
    })),

  setUnitCount: (count) =>
    set((state) => {
      const n = Math.max(1, Math.min(10, count));
      if (n === state.units.length) return state;
      if (n > state.units.length) {
        const last = state.units.at(-1);
        const toAdd = Array.from({ length: n - state.units.length }, () => ({
          ...defaultUnit(last?.unitType),
          enabled: true,
          useSeparateImage: last?.useSeparateImage ?? false,
        }));
        return { units: [...state.units, ...toAdd] };
      }
      return { units: state.units.slice(0, n) };
    }),

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

  setUnitType: (index, unitType) =>
    set((state) => {
      const units = [...state.units];
      const old = units[index];
      units[index] = { ...defaultUnit(unitType), enabled: old.enabled, useSeparateImage: old.useSeparateImage, image: old.image, images: old.images, masks: old.masks };
      return { units };
    }),

  toggleSeparateImage: (index) =>
    set((state) => {
      const units = [...state.units];
      const current = units[index].useSeparateImage;
      units[index] = {
        ...units[index],
        useSeparateImage: !current,
        // Clear image data when toggling OFF
        ...(!current ? {} : { image: null, processedImage: null }),
      };
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

  restoreUnits: (snapshots) =>
    set({
      units: snapshots.map((s) => ({
        enabled: s.enabled,
        unitType: s.unitType,
        useSeparateImage: s.useSeparateImage,
        processor: s.processor,
        model: s.model,
        mode: s.mode,
        strength: s.strength,
        start: s.start,
        end: s.end,
        image: s.image ? base64ToFile(s.image, "control.png") : null,
        processedImage: s.processedImage ? `data:image/png;base64,${s.processedImage}` : null,
        guess: s.guess,
        factor: s.factor,
        attention: s.attention,
        fidelity: s.fidelity,
        queryWeight: s.queryWeight,
        adainWeight: s.adainWeight,
        adapter: s.adapter,
        scale: s.scale,
        crop: s.crop,
        images: s.images.map((b, i) => base64ToFile(b, `ref-${i}.png`)),
        masks: s.masks.map((b, i) => base64ToFile(b, `mask-${i}.png`)),
        fitMode: s.fitMode ?? "contain",
      })),
    }),

  reset: () => set({ units: [defaultUnit()] }),
}));

function stripDataPrefix(dataUri: string): string {
  const idx = dataUri.indexOf(",");
  return idx >= 0 ? dataUri.slice(idx + 1) : dataUri;
}

/** Serialize all control units to JSON-safe snapshots (File → base64). */
export async function snapshotUnits(): Promise<ControlUnitSnapshot[]> {
  const { units } = useControlStore.getState();
  return Promise.all(
    units.map(async (u) => ({
      enabled: u.enabled,
      unitType: u.unitType,
      useSeparateImage: u.useSeparateImage,
      processor: u.processor,
      model: u.model,
      mode: u.mode,
      strength: u.strength,
      start: u.start,
      end: u.end,
      image: u.image ? await fileToBase64(u.image) : null,
      processedImage: u.processedImage ? stripDataPrefix(u.processedImage) : null,
      guess: u.guess,
      factor: u.factor,
      attention: u.attention,
      fidelity: u.fidelity,
      queryWeight: u.queryWeight,
      adainWeight: u.adainWeight,
      adapter: u.adapter,
      scale: u.scale,
      crop: u.crop,
      images: await Promise.all(u.images.map(fileToBase64)),
      masks: await Promise.all(u.masks.map(fileToBase64)),
      fitMode: u.fitMode,
    })),
  );
}

// --- IDB persistence (async File→base64 prevents using Zustand persist middleware) ---

const CONTROL_DB = "sdnext-control";
const CONTROL_STORE = "state";
const CONTROL_KEY = "units";

let controlDbPromise: Promise<IDBDatabase> | null = null;

function openControlDb(): Promise<IDBDatabase> {
  if (!controlDbPromise) {
    controlDbPromise = new Promise<IDBDatabase>((resolve, reject) => {
      const req = indexedDB.open(CONTROL_DB, 1);
      req.onupgradeneeded = () => {
        if (!req.result.objectStoreNames.contains(CONTROL_STORE)) {
          req.result.createObjectStore(CONTROL_STORE);
        }
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  }
  return controlDbPromise;
}

function readSnapshots(): Promise<ControlUnitSnapshot[] | null> {
  return openControlDb().then(
    (db) =>
      new Promise((resolve, reject) => {
        const tx = db.transaction(CONTROL_STORE, "readonly");
        const req = tx.objectStore(CONTROL_STORE).get(CONTROL_KEY);
        req.onsuccess = () => resolve((req.result as ControlUnitSnapshot[] | undefined) ?? null);
        req.onerror = () => reject(req.error);
      }),
  );
}

function writeSnapshots(snapshots: ControlUnitSnapshot[]): Promise<void> {
  return openControlDb().then(
    (db) =>
      new Promise((resolve, reject) => {
        const tx = db.transaction(CONTROL_STORE, "readwrite");
        tx.objectStore(CONTROL_STORE).put(snapshots, CONTROL_KEY);
        tx.oncomplete = () => resolve();
        tx.onerror = () => reject(tx.error);
      }),
  );
}

let controlPersistTimer: ReturnType<typeof setTimeout> | null = null;

function scheduleControlPersist() {
  if (controlPersistTimer) clearTimeout(controlPersistTimer);
  controlPersistTimer = setTimeout(() => {
    snapshotUnits().then(writeSnapshots);
  }, 2000);
}

// Subscribe to store changes → debounced IDB write
useControlStore.subscribe(scheduleControlPersist);

// Rehydrate on startup
readSnapshots().then((snapshots) => {
  if (snapshots && snapshots.length > 0) {
    useControlStore.getState().restoreUnits(snapshots);
  }
});
