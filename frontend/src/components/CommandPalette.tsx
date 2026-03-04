import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { matchSorter } from "match-sorter";
import { useShortcut } from "@/hooks/useShortcut";
import { useUiStore } from "@/stores/uiStore";
import { buildActions } from "@/lib/actionRegistry";
import { navigateToParam } from "@/lib/navigateToParam";
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
  const [search, setSearch] = useState("");
  const listRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    listRef.current?.scrollTo(0, 0);
  }, [search]);
  const recentIds = useUiStore((s) => s.recentCommandIds);
  const addRecentCommand = useUiStore((s) => s.addRecentCommand);

  useShortcut("command-palette", () => setOpen((o) => !o));

  const handleOpenChange = useCallback((next: boolean) => {
    setOpen(next);
    if (!next) setSearch("");
  }, []);

  const actions = useMemo(() => (open ? buildActions() : []), [open]);

  const recentActions = useMemo(() => {
    if (!open) return [];
    return recentIds.map((id) => actions.find((a) => a.id === id)).filter(Boolean) as typeof actions;
  }, [open, recentIds, actions]);

  const filteredActions = useMemo(() => {
    const isSearching = search.trim().length > 0;
    const base = isSearching ? actions : actions.filter((a) => !a.showOnlyInSearch);
    if (!isSearching) return base;
    return matchSorter(base, search, { keys: ["label", "keywords", "id"] });
  }, [actions, search]);

  const filteredRecent = useMemo(() => {
    if (!search.trim()) return recentActions;
    return matchSorter(recentActions, search, { keys: ["label", "keywords", "id"] });
  }, [recentActions, search]);

  const handleSelect = useCallback((actionId: string) => {
    const action = actions.find((a) => a.id === actionId);
    if (!action) return;
    setOpen(false);
    setSearch("");
    addRecentCommand(actionId);
    requestAnimationFrame(() => navigateToParam(action.target));
  }, [actions, addRecentCommand]);

  const grouped = useMemo(() => {
    const map = new Map<string, typeof actions>();
    for (const a of filteredActions) {
      if (a.group === "Recent") continue;
      if (!map.has(a.group)) map.set(a.group, []);
      map.get(a.group)!.push(a);
    }
    return map;
  }, [filteredActions]);

  return (
    <CommandDialog open={open} onOpenChange={handleOpenChange} shouldFilter={false} title="Command Palette" description="Search for a command to run..." showCloseButton={false}>
      <CommandInput placeholder="Type a command..." value={search} onValueChange={setSearch} />
      <CommandList ref={listRef}>
        <CommandEmpty>No results found.</CommandEmpty>

        {filteredRecent.length > 0 && (
          <CommandGroup heading="Recent">
            {filteredRecent.map((a) => (
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
    <CommandItem value={action.id} onSelect={() => onSelect(action.id)}>
      <Icon size={16} />
      <span>{action.label}</span>
      {shortcutDef && <CommandShortcut>{formatShortcut(shortcutDef)}</CommandShortcut>}
    </CommandItem>
  );
}
