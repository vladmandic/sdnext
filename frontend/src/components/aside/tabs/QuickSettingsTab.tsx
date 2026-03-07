import { useState, useMemo, useRef, useCallback } from "react";
import { useOptions, useSetOptions, useOptionsInfo } from "@/api/hooks/useSettings";
import { useModelList, useVaeList } from "@/api/hooks/useModels";
import { DEFAULT_QUICK_SETTINGS_KEYS, metaToSettingDef } from "@/lib/settingsSchema";
import type { SettingDef } from "@/lib/settingsSchema";
import { SettingControl } from "@/components/settings/SettingControl";
import { useUiStore } from "@/stores/uiStore";
import { Popover, PopoverTrigger, PopoverContent } from "@/components/ui/popover";
import { Command, CommandInput, CommandList, CommandEmpty, CommandGroup, CommandItem } from "@/components/ui/command";
import { Button } from "@/components/ui/button";
import { Pencil, RotateCcw, Check } from "lucide-react";

const SENTINEL_NONE = "None";
const SENTINEL_AUTOMATIC = "Automatic";

export function QuickSettingsTab() {
  const { data: options } = useOptions();
  const { data: optionsInfo } = useOptionsInfo();
  const setOptions = useSetOptions();
  const { data: models } = useModelList();
  const { data: vaes } = useVaeList();
  const debounceTimers = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map());
  const [pickerOpen, setPickerOpen] = useState(false);

  const customKeys = useUiStore((s) => s.quickSettingsKeys);
  const setCustomKeys = useUiStore((s) => s.setQuickSettingsKeys);
  const activeKeys = customKeys ?? DEFAULT_QUICK_SETTINGS_KEYS;

  const dynamicChoices = useMemo(() => {
    const choices: Record<string, string[]> = {};
    if (models) choices["sd_model_checkpoint"] = models.map((m) => m.title);
    if (models) choices["sd_model_refiner"] = [SENTINEL_NONE, ...models.map((m) => m.title)];
    if (vaes) choices["sd_vae"] = [SENTINEL_AUTOMATIC, SENTINEL_NONE, ...vaes.map((v) => v.name)];
    return choices;
  }, [models, vaes]);

  // Build groups from active keys, grouped by their backend section_title
  const groups = useMemo(() => {
    if (!optionsInfo || !options) return [];
    const meta = optionsInfo.options;
    const sectionMap = new Map<string, SettingDef[]>();
    for (const key of activeKeys) {
      const info = meta[key];
      if (!info) continue;
      if (!(key in options)) continue;
      const section = info.section_title || "Other";
      if (!sectionMap.has(section)) sectionMap.set(section, []);
      sectionMap.get(section)!.push(metaToSettingDef(key, info));
    }
    return Array.from(sectionMap.entries()).map(([title, settings]) => ({ title, settings }));
  }, [optionsInfo, options, activeKeys]);

  // All available settings for the picker (excluding separators and hidden)
  const allPickerItems = useMemo(() => {
    if (!optionsInfo || !options) return [];
    const meta = optionsInfo.options;
    const items: { key: string; label: string; section: string }[] = [];
    for (const [key, info] of Object.entries(meta)) {
      if (!info.visible || info.hidden || info.is_legacy) continue;
      if (info.component === "separator") continue;
      if (!(key in options)) continue;
      items.push({ key, label: info.label || key, section: info.section_title || "Other" });
    }
    return items;
  }, [optionsInfo, options]);

  // Group picker items by section
  const pickerSections = useMemo(() => {
    const sectionMap = new Map<string, { key: string; label: string }[]>();
    for (const item of allPickerItems) {
      if (!sectionMap.has(item.section)) sectionMap.set(item.section, []);
      sectionMap.get(item.section)!.push({ key: item.key, label: item.label });
    }
    return Array.from(sectionMap.entries()).map(([title, items]) => ({ title, items }));
  }, [allPickerItems]);

  const activeKeySet = useMemo(() => new Set(activeKeys), [activeKeys]);

  const toggleKey = useCallback((key: string) => {
    const current = customKeys ?? [...DEFAULT_QUICK_SETTINGS_KEYS];
    if (current.includes(key)) {
      setCustomKeys(current.filter((k) => k !== key));
    } else {
      setCustomKeys([...current, key]);
    }
  }, [customKeys, setCustomKeys]);

  const handleReset = useCallback(() => {
    setCustomKeys(null);
  }, [setCustomKeys]);

  const handleChange = useCallback((key: string, value: unknown, isSlider: boolean) => {
    if (isSlider) {
      const existing = debounceTimers.current.get(key);
      if (existing) clearTimeout(existing);
      debounceTimers.current.set(key, setTimeout(() => {
        setOptions.mutate({ [key]: value });
        debounceTimers.current.delete(key);
      }, 300));
    } else {
      setOptions.mutate({ [key]: value });
    }
  }, [setOptions]);

  if (!options || !optionsInfo) {
    return <div className="p-3 text-xs text-muted-foreground">Loading settings...</div>;
  }

  return (
    <div className="p-2 space-y-3">
      {/* Header with edit/reset controls */}
      <div className="flex items-center justify-end gap-1">
        {customKeys !== null && (
          <Button variant="ghost" size="icon" className="h-5 w-5" onClick={handleReset} title="Reset to defaults">
            <RotateCcw size={12} />
          </Button>
        )}
        <Popover open={pickerOpen} onOpenChange={setPickerOpen}>
          <PopoverTrigger asChild>
            <Button variant="ghost" size="icon" className="h-5 w-5" title="Edit quick settings">
              <Pencil size={12} />
            </Button>
          </PopoverTrigger>
          <PopoverContent className="w-72 p-0" align="end" side="left" sideOffset={8}>
            <Command>
              <CommandInput placeholder="Search settings..." className="text-xs h-8" />
              <CommandList className="max-h-72">
                <CommandEmpty>No settings found</CommandEmpty>
                {pickerSections.map((section) => (
                  <CommandGroup key={section.title} heading={section.title}>
                    {section.items.map((item) => (
                      <CommandItem key={item.key} value={`${item.label} ${item.key}`} onSelect={() => toggleKey(item.key)} className="text-xs gap-2 py-1">
                        <div className="flex items-center justify-center h-3.5 w-3.5 shrink-0 border border-border rounded-sm">
                          {activeKeySet.has(item.key) && <Check size={10} className="text-primary" />}
                        </div>
                        <span className="truncate">{item.label}</span>
                      </CommandItem>
                    ))}
                  </CommandGroup>
                ))}
              </CommandList>
              {customKeys !== null && (
                <div className="border-t border-border p-1.5">
                  <Button variant="ghost" size="sm" className="w-full text-xs h-6" onClick={handleReset}>
                    <RotateCcw size={12} />
                    Reset to defaults
                  </Button>
                </div>
              )}
            </Command>
          </PopoverContent>
        </Popover>
      </div>

      {/* Settings grouped by backend section */}
      {groups.map((group) => (
        <div key={group.title}>
          <h3 className="text-3xs font-semibold uppercase tracking-wider text-muted-foreground/60 mb-1.5">{group.title}</h3>
          <div className="space-y-1.5">
            {group.settings.map((setting) => (
              <div key={setting.key} className="flex items-center gap-2">
                <span className="text-2xs text-muted-foreground w-[6.875rem] shrink-0 truncate" title={setting.label}>{setting.label}</span>
                <div className="flex-1 min-w-0">
                  <SettingControl
                    setting={setting}
                    value={options[setting.key]}
                    onChange={(v) => handleChange(setting.key, v, setting.component === "slider")}
                    dynamicChoices={dynamicChoices[setting.key]}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      ))}

      {groups.length === 0 && (
        <div className="text-xs text-muted-foreground text-center py-4">
          No settings selected. Click <Pencil size={10} className="inline" /> to add settings.
        </div>
      )}
    </div>
  );
}
