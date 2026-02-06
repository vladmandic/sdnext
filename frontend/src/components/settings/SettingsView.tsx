import { useState, useMemo } from "react";
import { toast } from "sonner";
import { useOptions, useSetOptions } from "@/api/hooks/useSettings";
import { useModelList, useSamplerList, useVaeList, useUpscalerList } from "@/api/hooks/useModels";
import { settingsSchema, knownKeys } from "@/lib/settingsSchema";
import { SettingsSection } from "./SettingsSection";
import { SettingControl } from "./SettingControl";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { Save, RotateCcw, Search } from "lucide-react";

export function SettingsView() {
  const { data: options, isLoading } = useOptions();
  const setOptions = useSetOptions();
  const { data: models } = useModelList();
  const { data: samplers } = useSamplerList();
  const { data: vaes } = useVaeList();
  const { data: upscalers } = useUpscalerList();

  const [dirty, setDirty] = useState<Record<string, unknown>>({});
  const [activeSection, setActiveSection] = useState(settingsSchema[0].id);
  const [searchQuery, setSearchQuery] = useState("");

  const dynamicChoices = useMemo(() => {
    const choices: Record<string, string[]> = {};
    if (models) choices["sd_model_checkpoint"] = models.map((m) => m.title);
    if (models) choices["sd_model_refiner"] = ["None", ...models.map((m) => m.title)];
    if (vaes) choices["sd_vae"] = ["Automatic", "None", ...vaes.map((v) => v.model_name)];
    if (samplers) choices["sampler_name"] = samplers.map((s) => s.name);
    if (upscalers) choices["upscaler_for_img2img"] = upscalers.map((u) => u.name);
    return choices;
  }, [models, samplers, vaes, upscalers]);

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

  // Build advanced section from unknown keys
  const advancedSettings = useMemo(() => {
    if (!options) return [];
    return Object.entries(options)
      .filter(([key]) => !knownKeys.has(key))
      .filter(([key]) => !searchQuery || key.toLowerCase().includes(searchQuery.toLowerCase()))
      .sort(([a], [b]) => a.localeCompare(b));
  }, [options, searchQuery]);

  // Filter schema sections by search
  const filteredSections = useMemo(() => {
    if (!searchQuery) return settingsSchema;
    return settingsSchema
      .map((section) => ({
        ...section,
        settings: section.settings.filter(
          (s) =>
            s.key.toLowerCase().includes(searchQuery.toLowerCase()) ||
            s.label.toLowerCase().includes(searchQuery.toLowerCase()) ||
            (s.description?.toLowerCase().includes(searchQuery.toLowerCase()) ?? false),
        ),
      }))
      .filter((s) => s.settings.length > 0);
  }, [searchQuery]);

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
            {settingsSchema.map((section) => (
              <button
                key={section.id}
                onClick={() => {
                  setActiveSection(section.id);
                  setSearchQuery("");
                }}
                className={cn(
                  "text-left text-xs px-2 py-1.5 rounded-md transition-colors",
                  "hover:bg-accent hover:text-accent-foreground",
                  activeSection === section.id && !searchQuery && "bg-accent text-accent-foreground font-medium",
                )}
              >
                {section.title}
              </button>
            ))}
            <Separator className="my-1" />
            <button
              onClick={() => {
                setActiveSection("advanced");
                setSearchQuery("");
              }}
              className={cn(
                "text-left text-xs px-2 py-1.5 rounded-md transition-colors",
                "hover:bg-accent hover:text-accent-foreground",
                activeSection === "advanced" && !searchQuery && "bg-accent text-accent-foreground font-medium",
              )}
            >
              Advanced
            </button>
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
              {filteredSections.length === 0 && advancedSettings.length === 0 && (
                <p className="text-sm text-muted-foreground text-center py-8">No settings match your search</p>
              )}
            </div>
          ) : activeSection === "advanced" ? (
            // Advanced: all unknown settings
            <div className="space-y-4">
              <h2 className="text-sm font-semibold text-foreground">Advanced</h2>
              <p className="text-xs text-muted-foreground">
                Settings not covered by the schema above. Edit with care.
              </p>
              <div className="space-y-2">
                {advancedSettings.map(([key, val]) => {
                  const currentValue = dirty[key] ?? val;
                  const isDirty = key in dirty;
                  const inferredComponent = typeof val === "boolean" ? "switch" : typeof val === "number" ? "number" : "input";
                  return (
                    <div key={key} className="flex items-center gap-3">
                      <div className="w-56 flex-shrink-0">
                        <Label className="text-xs font-mono flex items-center gap-1.5 break-all">
                          {key}
                          {isDirty && <Badge variant="outline" className="text-[9px] px-1 py-0 h-3.5 text-primary border-primary/30">modified</Badge>}
                        </Label>
                      </div>
                      <div className="flex-1 min-w-0">
                        <SettingControl
                          setting={{ key, label: key, component: inferredComponent }}
                          value={currentValue}
                          onChange={(v) => handleSettingChange(key, v)}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          ) : (
            // Normal section view
            (() => {
              const section = filteredSections.find((s) => s.id === activeSection) ?? settingsSchema.find((s) => s.id === activeSection);
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
