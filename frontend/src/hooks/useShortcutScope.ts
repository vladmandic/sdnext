import { useEffect } from "react";
import { useShortcutStore } from "@/stores/shortcutStore";
import type { ShortcutScope } from "@/lib/shortcuts";

/**
 * Pushes a shortcut scope while mounted (and enabled).
 * Scoped shortcuts shadow lower scopes for the same key.
 */
export function useShortcutScope(scope: ShortcutScope, enabled = true) {
  useEffect(() => {
    if (!enabled) return;
    useShortcutStore.getState().pushScope(scope);
    return () => useShortcutStore.getState().popScope(scope);
  }, [scope, enabled]);
}
