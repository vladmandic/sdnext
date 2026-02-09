import type { SettingSectionDef } from "@/lib/settingsSchema";
import type { OptionsMap } from "@/api/types/settings";
import { SettingControl } from "./SettingControl";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";

interface SettingsSectionProps {
  section: SettingSectionDef;
  values: OptionsMap;
  dirty: Record<string, unknown>;
  onSettingChange: (key: string, value: unknown) => void;
  dynamicChoices?: Record<string, string[]>;
}

export function SettingsSection({ section, values, dirty, onSettingChange, dynamicChoices }: SettingsSectionProps) {
  return (
    <div className="space-y-4">
      <h2 className="text-sm font-semibold text-foreground">{section.title}</h2>
      <div className="space-y-3">
        {section.settings.map((setting) => {
          if (setting.component === "separator") {
            return (
              <div key={setting.key} className="pt-3 first:pt-0">
                <div className="border-b border-primary/30 pb-1">
                  <h3 className="text-sm font-semibold text-primary">{setting.label}</h3>
                </div>
              </div>
            );
          }
          const currentValue = dirty[setting.key] ?? values[setting.key] ?? setting.defaultValue;
          const isDirty = setting.key in dirty;
          return (
            <div key={setting.key} className="flex items-start gap-3">
              <div className="flex flex-col gap-0.5 w-48 flex-shrink-0 pt-1">
                <Label className="text-xs font-medium flex items-center gap-1.5">
                  {setting.label}
                  {isDirty && <Badge variant="outline" className="text-[9px] px-1 py-0 h-3.5 text-primary border-primary/30">modified</Badge>}
                  {setting.requiresRestart && <Badge variant="outline" className="text-[9px] px-1 py-0 h-3.5 text-amber-500 border-amber-500/30">restart</Badge>}
                </Label>
                {setting.description && (
                  <span className="text-[10px] text-muted-foreground leading-tight">{setting.description}</span>
                )}
              </div>
              <div className="flex-1 min-w-0">
                <SettingControl
                  setting={setting}
                  value={currentValue}
                  onChange={(v) => onSettingChange(setting.key, v)}
                  dynamicChoices={dynamicChoices?.[setting.key]}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
