import { useEffect } from "react";
import { useShortcutStore } from "@/stores/shortcutStore";
import { SHORTCUTS, matchesEvent } from "@/lib/shortcuts";

const SKIP_TAGS = new Set(["INPUT", "TEXTAREA", "SELECT"]);

/**
 * Single global keydown listener that dispatches to registered shortcut handlers.
 * Respects scope priority: topmost scope on the stack shadows lower scopes.
 * Skips events originating from form inputs.
 * Mount this once in AppShell.
 */
export function useShortcutDispatcher() {
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      // Allow modifier-key shortcuts (Ctrl+K, Ctrl+Enter) even inside inputs
      const hasModifier = e.ctrlKey || e.metaKey || e.altKey;
      if (!hasModifier && SKIP_TAGS.has(tag)) return;

      const { scopeStack, handlers } = useShortcutStore.getState();
      const activeScope = scopeStack[scopeStack.length - 1];

      // Try active scope first, then fall back to global
      const scopesToCheck = activeScope === "global" ? ["global"] : [activeScope, "global"];

      for (const scope of scopesToCheck) {
        for (const def of Object.values(SHORTCUTS)) {
          if (def.scope !== scope) continue;
          if (!matchesEvent(def, e)) continue;
          const handler = handlers.get(def.id);
          if (handler) {
            e.preventDefault();
            handler(e);
            return;
          }
        }
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);
}
