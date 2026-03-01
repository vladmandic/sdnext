import { useMemo } from "react";
import { useShortcutStore } from "@/stores/shortcutStore";
import { useShortcut } from "@/hooks/useShortcut";
import { SHORTCUTS, formatShortcut } from "@/lib/shortcuts";
import type { ShortcutCategory } from "@/lib/shortcuts";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";

export function ShortcutOverlay() {
  const open = useShortcutStore((s) => s.cheatsheetOpen);
  const setOpen = useShortcutStore((s) => s.setCheatsheetOpen);

  useShortcut("cheatsheet", () => setOpen(!open));

  const grouped = useMemo(() => {
    const map = new Map<ShortcutCategory, { label: string; keys: string }[]>();
    const seen = new Set<string>();
    for (const def of Object.values(SHORTCUTS)) {
      // Deduplicate display entries (e.g. canvas-delete & canvas-delete-backspace)
      const dedup = `${def.category}:${def.label}`;
      if (seen.has(dedup)) continue;
      seen.add(dedup);
      if (!map.has(def.category)) map.set(def.category, []);
      map.get(def.category)!.push({ label: def.label, keys: formatShortcut(def) });
    }
    return map;
  }, []);

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogContent className="sm:max-w-lg max-h-[80vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Keyboard Shortcuts</DialogTitle>
          <DialogDescription>Press ? to toggle this overlay</DialogDescription>
        </DialogHeader>
        <div className="flex flex-col gap-4 mt-2">
          {[...grouped.entries()].map(([category, shortcuts]) => (
            <div key={category}>
              <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">{category}</h3>
              <div className="flex flex-col gap-1">
                {shortcuts.map((s) => (
                  <div key={s.label} className="flex items-center justify-between py-1 px-2 rounded hover:bg-muted/50">
                    <span className="text-sm">{s.label}</span>
                    <kbd className="text-xs bg-muted px-1.5 py-0.5 rounded border border-border font-mono">{s.keys}</kbd>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </DialogContent>
    </Dialog>
  );
}
