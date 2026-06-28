import { gradioApp, onAfterUiUpdate } from './script';

// Aspect-ratio lock for the paired width/height sliders. The math runs client-side and debounced,
// and only the partner axis is ever written, never the field being edited; writing the edited field
// back is what yanked the value mid-type when this went through a gradio round-trip. Programmatic
// updates dispatch a synthetic input event so gradio's store stays in sync. State is keyed per
// dropdown element so duplicate elem ids across tabs stay isolated.

const RES_DEBOUNCE = 350;
const AR_DEBOUNCE = 120;
const timers = new WeakMap<Element, ReturnType<typeof setTimeout>>();
const busy = new WeakSet<Element>();

function parseAR(ar: string): [number, number] | null {
  if (!ar || ar === 'AR') return null;
  const parts = ar.split(':');
  if (parts.length !== 2) return null;
  const w = parseInt(parts[0], 10);
  const h = parseInt(parts[1], 10);
  return (w > 0 && h > 0) ? [w, h] : null;
}

function numberInput(group: Element): HTMLInputElement | null {
  const inp = group.querySelector('input[type=number]') || group.querySelector('input');
  return inp instanceof HTMLInputElement ? inp : null;
}

function readValue(group: Element): number {
  const inp = numberInput(group);
  return inp ? Number(inp.value) : 0;
}

function writeValue(group: Element, raw: number): void {
  const inp = numberInput(group);
  if (!inp) return;
  const step = Number(inp.step) || 8;
  const min = inp.min !== '' ? Number(inp.min) : 0;
  const max = inp.max !== '' ? Number(inp.max) : 8192;
  const value = Math.max(min, Math.min(max, Math.round(raw / step) * step));
  if (value === Number(inp.value)) return; // unchanged: skip so the listeners do not refire
  group.querySelectorAll('input').forEach((el) => {
    if (!(el instanceof HTMLInputElement)) return;
    el.value = String(value);
    const e = new Event('input', { bubbles: true });
    Object.defineProperty(e, 'target', { value: el });
    el.dispatchEvent(e);
  });
}

function arValue(arEl: Element): string {
  const inp = arEl.querySelector('input');
  return inp instanceof HTMLInputElement ? inp.value : 'AR';
}

function pairOf(arEl: Element): { width: Element; height: Element } | null {
  let container: Element | null = arEl.parentElement;
  for (let i = 0; i < 6 && container; i++) {
    const width = container.querySelector('[id$="_width"]');
    const height = container.querySelector('[id$="_height"]');
    if (width && height) return { width, height };
    container = container.parentElement;
  }
  return null;
}

function settle(arEl: Element, source: 'width' | 'height'): void {
  const ar = parseAR(arValue(arEl));
  if (!ar) return;
  const pair = pairOf(arEl);
  if (!pair) return;
  const [rw, rh] = ar;
  busy.add(arEl);
  if (source === 'height') writeValue(pair.width, (readValue(pair.height) * rw) / rh);
  else writeValue(pair.height, (readValue(pair.width) * rh) / rw);
  busy.delete(arEl);
}

function schedule(arEl: Element, source: 'width' | 'height', delay: number): void {
  if (busy.has(arEl)) return; // ignore the input events our own writes dispatch
  clearTimeout(timers.get(arEl));
  timers.set(arEl, setTimeout(() => settle(arEl, source), delay));
}

function flush(arEl: Element, source: 'width' | 'height'): void {
  if (busy.has(arEl)) return;
  clearTimeout(timers.get(arEl));
  settle(arEl, source);
}

function bind(arEl: Element, group: Element, source: 'width' | 'height'): void {
  group.querySelectorAll('input').forEach((el) => {
    if (!(el instanceof HTMLInputElement) || el.classList.contains('ar-lock-bound')) return;
    el.classList.add('ar-lock-bound');
    el.addEventListener('input', () => schedule(arEl, source, RES_DEBOUNCE));
    el.addEventListener('change', () => flush(arEl, source)); // commit on blur, enter, or slider release
  });
}

export function setupResolutionLock(): void {
  gradioApp().querySelectorAll('.ar-dropdown').forEach((arEl) => {
    const pair = pairOf(arEl);
    if (!pair) return;
    bind(arEl, pair.width, 'width');
    bind(arEl, pair.height, 'height');
    arEl.querySelectorAll('input').forEach((el) => {
      if (!(el instanceof HTMLInputElement) || el.classList.contains('ar-lock-bound')) return;
      el.classList.add('ar-lock-bound');
      el.addEventListener('change', () => flush(arEl, 'width')); // new ratio: keep width, derive height
      el.addEventListener('input', () => schedule(arEl, 'width', AR_DEBOUNCE));
    });
  });
}

onAfterUiUpdate(setupResolutionLock);
