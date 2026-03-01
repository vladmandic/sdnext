export type ShortcutScope = "global" | "canvas" | "gallery" | "lightbox" | "comparison";

export type ShortcutCategory = "Global" | "Canvas" | "Lightbox" | "Comparison" | "Navigation";

export interface ShortcutDef {
  id: string;
  key: string;
  ctrl?: boolean;
  shift?: boolean;
  alt?: boolean;
  meta?: boolean;
  scope: ShortcutScope;
  category: ShortcutCategory;
  label: string;
}

export const SHORTCUTS: Record<string, ShortcutDef> = {
  // Global
  "command-palette": { id: "command-palette", key: "k", ctrl: true, scope: "global", category: "Global", label: "Command palette" },
  "cheatsheet": { id: "cheatsheet", key: "?", scope: "global", category: "Global", label: "Keyboard shortcuts" },
  "generate": { id: "generate", key: "Enter", ctrl: true, scope: "global", category: "Global", label: "Generate" },
  "skip": { id: "skip", key: "Enter", ctrl: true, shift: true, scope: "global", category: "Global", label: "Skip current step" },
  "toggle-sidebar": { id: "toggle-sidebar", key: "b", ctrl: true, scope: "global", category: "Global", label: "Toggle sidebar" },
  "toggle-left-panel": { id: "toggle-left-panel", key: "\\", ctrl: true, scope: "global", category: "Global", label: "Toggle left panel" },
  "toggle-right-panel": { id: "toggle-right-panel", key: "\\", ctrl: true, shift: true, scope: "global", category: "Global", label: "Toggle right panel" },

  // Canvas
  "canvas-move": { id: "canvas-move", key: "v", scope: "canvas", category: "Canvas", label: "Move tool" },
  "canvas-brush": { id: "canvas-brush", key: "b", scope: "canvas", category: "Canvas", label: "Brush tool" },
  "canvas-eraser": { id: "canvas-eraser", key: "e", scope: "canvas", category: "Canvas", label: "Eraser tool" },
  "canvas-deselect": { id: "canvas-deselect", key: "Escape", scope: "canvas", category: "Canvas", label: "Deselect / move" },
  "canvas-brush-smaller": { id: "canvas-brush-smaller", key: "[", scope: "canvas", category: "Canvas", label: "Smaller brush" },
  "canvas-brush-larger": { id: "canvas-brush-larger", key: "]", scope: "canvas", category: "Canvas", label: "Larger brush" },
  "canvas-delete": { id: "canvas-delete", key: "Delete", scope: "canvas", category: "Canvas", label: "Remove layer" },
  "canvas-delete-backspace": { id: "canvas-delete-backspace", key: "Backspace", scope: "canvas", category: "Canvas", label: "Remove layer" },

  // Lightbox
  "lightbox-close": { id: "lightbox-close", key: "Escape", scope: "lightbox", category: "Lightbox", label: "Close lightbox" },
  "lightbox-prev": { id: "lightbox-prev", key: "ArrowLeft", scope: "lightbox", category: "Lightbox", label: "Previous image" },
  "lightbox-next": { id: "lightbox-next", key: "ArrowRight", scope: "lightbox", category: "Lightbox", label: "Next image" },
  "lightbox-zoom-in": { id: "lightbox-zoom-in", key: "+", scope: "lightbox", category: "Lightbox", label: "Zoom in" },
  "lightbox-zoom-in-eq": { id: "lightbox-zoom-in-eq", key: "=", scope: "lightbox", category: "Lightbox", label: "Zoom in" },
  "lightbox-zoom-out": { id: "lightbox-zoom-out", key: "-", scope: "lightbox", category: "Lightbox", label: "Zoom out" },
  "lightbox-zoom-reset": { id: "lightbox-zoom-reset", key: "0", scope: "lightbox", category: "Lightbox", label: "Reset zoom" },

  // Comparison
  "comparison-close": { id: "comparison-close", key: "Escape", scope: "comparison", category: "Comparison", label: "Close comparison" },
  "comparison-side-by-side": { id: "comparison-side-by-side", key: "1", scope: "comparison", category: "Comparison", label: "Side-by-side mode" },
  "comparison-swipe": { id: "comparison-swipe", key: "2", scope: "comparison", category: "Comparison", label: "Swipe mode" },
  "comparison-overlay": { id: "comparison-overlay", key: "3", scope: "comparison", category: "Comparison", label: "Overlay mode" },
  "comparison-diff": { id: "comparison-diff", key: "4", scope: "comparison", category: "Comparison", label: "Diff mode" },
  "comparison-toggle": { id: "comparison-toggle", key: " ", scope: "comparison", category: "Comparison", label: "Toggle A/B (overlay)" },
  "comparison-swap": { id: "comparison-swap", key: "s", scope: "comparison", category: "Comparison", label: "Swap images" },
  "comparison-zoom-in": { id: "comparison-zoom-in", key: "=", scope: "comparison", category: "Comparison", label: "Zoom in" },
  "comparison-zoom-out": { id: "comparison-zoom-out", key: "-", scope: "comparison", category: "Comparison", label: "Zoom out" },
  "comparison-zoom-reset": { id: "comparison-zoom-reset", key: "0", scope: "comparison", category: "Comparison", label: "Reset zoom" },
};

export function matchesEvent(def: ShortcutDef, e: KeyboardEvent): boolean {
  const ctrl = def.ctrl ?? false;
  const shift = def.shift ?? false;
  const alt = def.alt ?? false;
  const meta = def.meta ?? false;

  if (e.ctrlKey !== ctrl || e.shiftKey !== shift || e.altKey !== alt || e.metaKey !== meta) {
    return false;
  }

  // For "?" the key is literally "?" but KeyboardEvent.key reports "?" when Shift+/ is pressed
  if (e.key === def.key) return true;
  // Case-insensitive letter match (e.key is "B" when shift is held, but we compare lowercase for unmodified keys)
  if (def.key.length === 1 && e.key.toLowerCase() === def.key.toLowerCase() && !ctrl && !alt && !meta) return true;

  return false;
}

const MOD_SYMBOL = navigator.platform.includes("Mac") ? "\u2318" : "Ctrl";
const SHIFT_SYMBOL = navigator.platform.includes("Mac") ? "\u21e7" : "Shift";
const ALT_SYMBOL = navigator.platform.includes("Mac") ? "\u2325" : "Alt";

const KEY_LABELS: Record<string, string> = {
  Enter: "\u23ce",
  Escape: "Esc",
  ArrowLeft: "\u2190",
  ArrowRight: "\u2192",
  ArrowUp: "\u2191",
  ArrowDown: "\u2193",
  Backspace: "\u232b",
  Delete: "Del",
  " ": "Space",
  "\\": "\\",
};

export function formatShortcut(def: ShortcutDef): string {
  const parts: string[] = [];
  if (def.ctrl) parts.push(MOD_SYMBOL);
  if (def.shift) parts.push(SHIFT_SYMBOL);
  if (def.alt) parts.push(ALT_SYMBOL);
  if (def.meta && !navigator.platform.includes("Mac")) parts.push("Meta");
  parts.push(KEY_LABELS[def.key] ?? def.key.toUpperCase());
  return parts.join("+");
}
