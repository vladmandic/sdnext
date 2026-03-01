import { useCallback, useMemo, useState } from "react";
import { useShortcut } from "@/hooks/useShortcut";
import { useUiStore } from "@/stores/uiStore";
import { buildActions } from "@/lib/actionRegistry";
import { SHORTCUTS, formatShortcut } from "@/lib/shortcuts";
import {
  CommandDialog,
  CommandInput,
  CommandList,
  CommandEmpty,
  CommandGroup,
  CommandItem,
  CommandShortcut,
} from "@/components/ui/command";

export function CommandPalette() {
  const [open, setOpen] = useState(false);
  const recentIds = useUiStore((s) => s.recentCommandIds);
  const addRecentCommand = useUiStore((s) => s.addRecentCommand);

  useShortcut("command-palette", () => setOpen((o) => !o));

  const actions = useMemo(() => (open ? buildActions() : []), [open]);

  const recentActions = useMemo(() => {
    if (!open) return [];
    return recentIds.map((id) => actions.find((a) => a.id === id)).filter(Boolean) as typeof actions;
  }, [open, recentIds, actions]);

  const handleSelect = useCallback((actionId: string) => {
    const action = actions.find((a) => a.id === actionId);
    if (!action) return;
    setOpen(false);
    addRecentCommand(actionId);
    // Defer action so the dialog closes first
    requestAnimationFrame(() => action.action());
  }, [actions, addRecentCommand]);

  const grouped = useMemo(() => {
    const map = new Map<string, typeof actions>();
    for (const a of actions) {
      if (a.group === "Recent") continue;
      if (!map.has(a.group)) map.set(a.group, []);
      map.get(a.group)!.push(a);
    }
    return map;
  }, [actions]);

  return (
    <CommandDialog open={open} onOpenChange={setOpen} title="Command Palette" description="Search for a command to run..." showCloseButton={false}>
      <CommandInput placeholder="Type a command..." />
      <CommandList>
        <CommandEmpty>No results found.</CommandEmpty>

        {recentActions.length > 0 && (
          <CommandGroup heading="Recent">
            {recentActions.map((a) => (
              <PaletteItem key={`recent-${a.id}`} action={a} onSelect={handleSelect} />
            ))}
          </CommandGroup>
        )}

        {[...grouped.entries()].map(([group, items]) => (
          <CommandGroup key={group} heading={group}>
            {items.map((a) => (
              <PaletteItem key={a.id} action={a} onSelect={handleSelect} />
            ))}
          </CommandGroup>
        ))}
      </CommandList>
    </CommandDialog>
  );
}

function PaletteItem({ action, onSelect }: { action: ReturnType<typeof buildActions>[number]; onSelect: (id: string) => void }) {
  const Icon = action.icon;
  const shortcutDef = action.shortcutId ? SHORTCUTS[action.shortcutId] : undefined;

  return (
    <CommandItem value={`${action.id} ${action.label} ${action.keywords.join(" ")}`} onSelect={() => onSelect(action.id)}>
      <Icon size={16} />
      <span>{action.label}</span>
      {shortcutDef && <CommandShortcut>{formatShortcut(shortcutDef)}</CommandShortcut>}
    </CommandItem>
  );
}
