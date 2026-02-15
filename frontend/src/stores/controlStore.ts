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
    imageSource: "canvas",
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

/** Resolve the actual image for a control unit, following "unit:N" references. */
export function resolveUnitImage(units: ControlUnit[], index: number): File | null {
  const unit = units[index];
  if (!unit) return null;
  if (unit.imageSource === "separate") return unit.image;
  const match = unit.imageSource.match(/^unit:(\d+)$/);
  if (match) return units[Number(match[1])]?.image ?? null;
  return null; // "canvas" → no explicit image
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
  setImageSource: (index: number, source: string) => void;
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
      };
      return { units: [...state.units, newUnit] };
    }),

  removeUnit: (index) =>
    set((state) => {
      if (state.units.length <= 1) return state;
      const remaining = state.units.filter((_, i) => i !== index);
      // Cascade: fix or reset imageSource references
      const fixed = remaining.map((u) => {
        const match = u.imageSource.match(/^unit:(\d+)$/);
        if (!match) return u;
        const ref = Number(match[1]);
        if (ref === index) return { ...u, imageSource: "canvas" };
        if (ref > index) return { ...u, imageSource: `unit:${ref - 1}` };
        return u;
      });
      return { units: fixed };
    }),

  setUnitCount: (count) =>
    set((state) => {
      const n = Math.max(1, Math.min(10, count));
      if (n === state.units.length) return state;
      if (n > state.units.length) {
        const last = state.units.at(-1);
        const toAdd = Array.from({ length: n - state.units.length }, () => ({
          ...defaultUnit(last?.unitType),
          enabled: true,
        }));
        return { units: [...state.units, ...toAdd] };
      }
      // When shrinking, fix references pointing to removed units
      const kept = state.units.slice(0, n);
      const fixed = kept.map((u) => {
        const match = u.imageSource.match(/^unit:(\d+)$/);
        if (match && Number(match[1]) >= n) return { ...u, imageSource: "canvas" };
        return u;
      });
      return { units: fixed };
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
      units[index] = { ...defaultUnit(unitType), enabled: old.enabled, imageSource: old.imageSource, image: old.image, images: old.images, masks: old.masks };
      return { units };
    }),

  setImageSource: (index, source) =>
    set((state) => {
      const units = [...state.units];
      const old = units[index];
      const wasSeparate = old.imageSource === "separate";
      const isSeparate = source === "separate";
      units[index] = {
        ...old,
        imageSource: source,
        // Clear own image/processed when leaving "separate"
        ...(wasSeparate && !isSeparate ? { image: null, processedImage: null } : {}),
      };
      // If this unit stopped being "separate", cascade: reset anyone referencing it
      if (wasSeparate && !isSeparate) {
        for (let i = 0; i < units.length; i++) {
          if (units[i].imageSource === `unit:${index}`) {
            units[i] = { ...units[i], imageSource: "canvas" };
          }
        }
      }
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
      units: snapshots.map((s) => {
        // Migrate old snapshots: useSeparateImage boolean → imageSource string
        const raw = s as ControlUnitSnapshot & { useSeparateImage?: boolean };
        let imageSource: string;
        if (typeof s.imageSource === "string") {
          imageSource = s.imageSource;
        } else if (typeof raw.useSeparateImage === "boolean") {
          imageSource = raw.useSeparateImage ? "separate" : "canvas";
        } else {
          imageSource = "canvas";
        }
        return {
          enabled: s.enabled,
          unitType: s.unitType,
          imageSource,
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
        };
      }),
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
      imageSource: u.imageSource,
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
