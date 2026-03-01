import { useCallback, useEffect, useRef } from "react";
import { useShortcutStore } from "@/stores/shortcutStore";

/**
 * Registers a keyboard shortcut handler that is active while the component is mounted.
 * The `id` must match a key in the SHORTCUTS map defined in shortcuts.ts.
 */
export function useShortcut(id: string, handler: (e: KeyboardEvent) => void, enabled = true) {
  const handlerRef = useRef(handler);
  useEffect(() => { handlerRef.current = handler; });

  const stableHandler = useCallback((e: KeyboardEvent) => handlerRef.current(e), []);

  useEffect(() => {
    if (!enabled) return;
    useShortcutStore.getState().register(id, stableHandler);
    return () => useShortcutStore.getState().unregister(id);
  }, [id, enabled, stableHandler]);
}
