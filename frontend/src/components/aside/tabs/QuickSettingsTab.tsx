import { useMemo, useRef, useCallback } from "react";
import { useOptions, useSetOptions, useOptionsInfo } from "@/api/hooks/useSettings";
import { useModelList, useVaeList } from "@/api/hooks/useModels";
import { QUICK_SETTINGS_GROUPS, metaToSettingDef } from "@/lib/settingsSchema";
import type { SettingDef } from "@/lib/settingsSchema";
import { SettingControl } from "@/components/settings/SettingControl";

const SENTINEL_NONE = "None";
const SENTINEL_AUTOMATIC = "Automatic";

export function QuickSettingsTab() {
  const { data: options } = useOptions();
  const { data: optionsInfo } = useOptionsInfo();
  const setOptions = useSetOptions();
  const { data: models } = useModelList();
  const { data: vaes } = useVaeList();
  const debounceTimers = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map());

  const dynamicChoices = useMemo(() => {
    const choices: Record<string, string[]> = {};
    if (models) choices["sd_model_checkpoint"] = models.map((m) => m.title);
    if (models) choices["sd_model_refiner"] = [SENTINEL_NONE, ...models.map((m) => m.title)];
    if (vaes) choices["sd_vae"] = [SENTINEL_AUTOMATIC, SENTINEL_NONE, ...vaes.map((v) => v.model_name)];
    return choices;
  }, [models, vaes]);

  const groups = useMemo(() => {
    if (!optionsInfo || !options) return [];
    const meta = optionsInfo.options;
    return QUICK_SETTINGS_GROUPS.map((group) => {
      const settings: SettingDef[] = [];
      for (const key of group.keys) {
        const info = meta[key];
        if (!info) continue;
        if (!(key in options)) continue;
        settings.push(metaToSettingDef(key, info));
      }
      return { title: group.title, settings };
    }).filter((g) => g.settings.length > 0);
  }, [optionsInfo, options]);

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
      {groups.map((group) => (
        <div key={group.title}>
          <h3 className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground/60 mb-1.5">{group.title}</h3>
          <div className="space-y-1.5">
            {group.settings.map((setting) => (
              <div key={setting.key} className="flex items-center gap-2">
                <span className="text-[11px] text-muted-foreground w-[110px] shrink-0 truncate" title={setting.label}>{setting.label}</span>
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
    </div>
  );
}
