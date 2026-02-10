import { useState, useMemo } from "react";
import { toast } from "sonner";
import { useOptions, useSetOptions, useOptionsInfo } from "@/api/hooks/useSettings";
import { useModelList, useSamplerList, useVaeList, useUpscalerList } from "@/api/hooks/useModels";
import type { OptionInfoMeta } from "@/api/types/settings";
import type { SettingSectionDef, SettingDef } from "@/lib/settingsSchema";
import { settingsSchema, getSettingsMap, metaToSettingDef } from "@/lib/settingsSchema";
import { SettingsSection } from "./SettingsSection";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import { Save, RotateCcw, Search } from "lucide-react";

/** Build a SettingDef: backend metadata is authoritative for choices, curated provides label/description/component overrides */
function buildSettingDef(
  key: string,
  info: OptionInfoMeta,
  curatedMap: Map<string, { section: SettingSectionDef; setting: SettingDef }>,
): SettingDef {
  const curated = curatedMap.get(key);
  if (!curated) return metaToSettingDef(key, info);
  const base = metaToSettingDef(key, info);
  return {
    ...base,
    label: curated.setting.label,
    ...(curated.setting.description !== undefined && { description: curated.setting.description }),
    component: curated.setting.component,
    ...(curated.setting.min !== undefined && { min: curated.setting.min }),
    ...(curated.setting.max !== undefined && { max: curated.setting.max }),
    ...(curated.setting.step !== undefined && { step: curated.setting.step }),
    ...(curated.setting.requiresRestart && { requiresRestart: true }),
    // choices: prefer backend (base), fall back to curated if backend has none
    choices: base.choices ?? curated.setting.choices,
  };
}

export function SettingsView() {
  const { data: options, isLoading } = useOptions();
  const setOptions = useSetOptions();
  const { data: optionsInfo } = useOptionsInfo();
  const { data: models } = useModelList();
  const { data: samplers } = useSamplerList();
  const { data: vaes } = useVaeList();
  const { data: upscalers } = useUpscalerList();

  const [dirty, setDirty] = useState<Record<string, unknown>>({});
  const [activeSection, setActiveSection] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");

  const curatedMap = useMemo(() => getSettingsMap(), []);

  const dynamicChoices = useMemo(() => {
    const choices: Record<string, string[]> = {};
    if (models) choices["sd_model_checkpoint"] = models.map((m) => m.title);
    if (models) choices["sd_model_refiner"] = ["None", ...models.map((m) => m.title)];
    if (vaes) choices["sd_vae"] = ["Automatic", "None", ...vaes.map((v) => v.model_name)];
    if (samplers) choices["sampler_name"] = samplers.map((s) => s.name);
    if (upscalers) choices["upscaler_for_img2img"] = upscalers.map((u) => u.name);
    return choices;
  }, [models, samplers, vaes, upscalers]);

  // Build all sections from backend metadata, with curated overrides
  const allSections = useMemo((): SettingSectionDef[] => {
    if (!optionsInfo || !options) return settingsSchema; // fallback before metadata loads

    const meta = optionsInfo.options;
    const result: SettingSectionDef[] = [];

    for (const section of optionsInfo.sections) {
      if (section.hidden) continue;

      const settings: SettingDef[] = [];
      for (const [key, info] of Object.entries(meta)) {
        if (info.section_id !== section.id) continue;
        if (!info.visible || info.hidden || info.is_legacy) continue;
        if (info.component === "separator") {
          if (info.label) settings.push({ key, label: info.label, component: "separator" });
          continue;
        }
        if (!(key in options)) continue;

        settings.push(buildSettingDef(key, info, curatedMap));
      }

      if (settings.length > 0) {
        result.push({ id: section.id, title: section.title, settings });
      }
    }

    return result;
  }, [optionsInfo, options, curatedMap]);

  // Resolve active section: use first section if none selected or selection is stale
  const resolvedActive = useMemo(() => {
    if (allSections.length === 0) return null;
    if (activeSection && allSections.some((s) => s.id === activeSection)) return activeSection;
    return allSections[0].id;
  }, [allSections, activeSection]);

  // Filter sections by search
  const filteredSections = useMemo(() => {
    if (!searchQuery) return allSections;
    const q = searchQuery.toLowerCase();
    return allSections
      .map((section) => ({
        ...section,
        settings: section.settings.filter(
          (s) =>
            s.key.toLowerCase().includes(q) ||
            s.label.toLowerCase().includes(q) ||
            (s.description?.toLowerCase().includes(q) ?? false) ||
            section.title.toLowerCase().includes(q),
        ),
      }))
      .filter((s) => s.settings.length > 0);
  }, [allSections, searchQuery]);

  function handleSettingChange(key: string, value: unknown) {
    setDirty((prev) => ({ ...prev, [key]: value }));
  }

  async function handleApply() {
    if (Object.keys(dirty).length === 0) return;
    try {
      await setOptions.mutateAsync(dirty);
      toast.success("Settings saved", { description: `${Object.keys(dirty).length} setting(s) updated` });
      setDirty({});
    } catch (err) {
      toast.error("Failed to save settings", { description: err instanceof Error ? err.message : String(err) });
    }
  }

  function handleReset() {
    setDirty({});
  }

  const dirtyCount = Object.keys(dirty).length;

  if (isLoading || !options) {
    return (
      <div className="flex items-center justify-center h-full text-muted-foreground text-sm">
        Loading settings...
      </div>
    );
  }

  return (
    <div className="flex h-full">
      {/* Section navigation */}
      <div className="w-48 border-r border-border flex-shrink-0 flex flex-col">
        <div className="p-2">
          <div className="relative">
            <Search size={14} className="absolute left-2 top-1/2 -translate-y-1/2 text-muted-foreground" />
            <Input
              placeholder="Search..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="h-7 text-xs pl-7"
            />
          </div>
        </div>
        <ScrollArea className="flex-1">
          <div className="flex flex-col gap-0.5 p-1">
            {allSections.map((section) => (
              <button
                key={section.id}
                onClick={() => {
                  setActiveSection(section.id);
                  setSearchQuery("");
                }}
                className={cn(
                  "text-left text-xs px-2 py-1.5 rounded-md transition-colors",
                  "hover:bg-accent hover:text-accent-foreground",
                  resolvedActive === section.id && !searchQuery && "bg-accent text-accent-foreground font-medium",
                )}
              >
                {section.title}
              </button>
            ))}
          </div>
        </ScrollArea>

        {/* Apply / Reset */}
        <div className="p-2 border-t border-border flex flex-col gap-1.5">
          <Button
            size="sm"
            onClick={handleApply}
            disabled={dirtyCount === 0 || setOptions.isPending}
            className="w-full text-xs"
          >
            <Save size={14} />
            Apply{dirtyCount > 0 && ` (${dirtyCount})`}
          </Button>
          <Button
            variant="secondary"
            size="sm"
            onClick={handleReset}
            disabled={dirtyCount === 0}
            className="w-full text-xs"
          >
            <RotateCcw size={14} />
            Reset
          </Button>
        </div>
      </div>

      {/* Settings content */}
      <ScrollArea className="flex-1">
        <div className="p-4 max-w-2xl">
          {searchQuery ? (
            // Search results across all sections
            <div className="space-y-6">
              {filteredSections.map((section) => (
                <SettingsSection
                  key={section.id}
                  section={section}
                  values={options}
                  dirty={dirty}
                  onSettingChange={handleSettingChange}
                  dynamicChoices={dynamicChoices}
                />
              ))}
              {filteredSections.length === 0 && (
                <p className="text-sm text-muted-foreground text-center py-8">No settings match your search</p>
              )}
            </div>
          ) : (
            // Single section view
            (() => {
              const section = allSections.find((s) => s.id === resolvedActive);
              if (!section) return null;
              return (
                <SettingsSection
                  section={section}
                  values={options}
                  dirty={dirty}
                  onSettingChange={handleSettingChange}
                  dynamicChoices={dynamicChoices}
                />
              );
            })()
          )}
        </div>
      </ScrollArea>
    </div>
  );
}
