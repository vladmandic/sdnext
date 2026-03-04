import { useCallback } from "react";
import type { SettingSectionDef } from "@/lib/settingsSchema";
import type { OptionsMap } from "@/api/types/settings";
import { SettingControl } from "./SettingControl";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Tooltip, TooltipTrigger, TooltipContent } from "@/components/ui/tooltip";
import { getParamHelp } from "@/data/parameterHelp";
import { cn } from "@/lib/utils";
import { highlightMatch } from "@/lib/highlightMatch";
import { RotateCcw } from "lucide-react";

interface SettingsSectionProps {
  section: SettingSectionDef;
  values: OptionsMap;
  dirty: Record<string, unknown>;
  onSettingChange: (key: string, value: unknown) => void;
  dynamicChoices?: Record<string, string[]>;
  searchQuery?: string;
  onNavigateToSection?: (id: string) => void;
}

export function SettingsSection({ section, values, dirty, onSettingChange, dynamicChoices, searchQuery, onNavigateToSection }: SettingsSectionProps) {
  const getSettingValue = useCallback(
    (key: string) => dirty[key] ?? values[key],
    [dirty, values],
  );

  return (
    <div className="space-y-4">
      {searchQuery && onNavigateToSection ? (
        <button
          type="button"
          onClick={() => onNavigateToSection(section.id)}
          className="text-sm font-semibold text-primary hover:underline text-left"
        >
          {section.title} <span className="text-3xs font-normal text-muted-foreground">(go to section)</span>
        </button>
      ) : (
        <h2 className="text-sm font-semibold text-foreground">{section.title}</h2>
      )}
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
          const inline = setting.component === "switch" || setting.component === "number" || setting.component === "color";
          const helpText = getParamHelp(setting.label);
          const labelEl = (
            <Label className={cn("text-xs font-medium flex items-center gap-1.5", helpText && "cursor-help")}>
              {highlightMatch(setting.label, searchQuery ?? "")}
              {isDirty && <Badge variant="outline" className="text-4xs px-1 py-0 h-3.5 text-primary border-primary/30">modified</Badge>}
              {isDirty && setting.defaultValue !== undefined && (
                <button
                  onClick={() => onSettingChange(setting.key, setting.defaultValue)}
                  className="text-muted-foreground hover:text-primary transition-colors"
                  title="Restore default"
                >
                  <RotateCcw size={10} />
                </button>
              )}
              {setting.requiresRestart && <Badge variant="outline" className="text-4xs px-1 py-0 h-3.5 text-amber-500 border-amber-500/30">restart</Badge>}
            </Label>
          );
          const labelBlock = helpText ? (
            <Tooltip>
              <TooltipTrigger asChild>{labelEl}</TooltipTrigger>
              <TooltipContent><span dangerouslySetInnerHTML={{ __html: helpText }} /></TooltipContent>
            </Tooltip>
          ) : labelEl;
          const controlBlock = (
            <SettingControl
              setting={setting}
              value={currentValue}
              onChange={(v) => onSettingChange(setting.key, v)}
              dynamicChoices={dynamicChoices?.[setting.key]}
              getSettingValue={getSettingValue}
            />
          );
          const indented = !!setting.baseFolderKey;
          return inline ? (
            <div key={setting.key} className={cn("flex items-center justify-between gap-3", indented && "pl-4")}>
              {labelBlock}
              {controlBlock}
            </div>
          ) : (
            <div key={setting.key} className={cn("flex flex-col gap-1", indented && "pl-4")}>
              {labelBlock}
              {controlBlock}
            </div>
          );
        })}
      </div>
    </div>
  );
}
