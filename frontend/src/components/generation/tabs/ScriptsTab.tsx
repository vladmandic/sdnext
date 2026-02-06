import { useMemo } from "react";
import { useScriptsList, useScriptInfo } from "@/api/hooks/useScripts";
import { useScriptStore } from "@/stores/scriptStore";
import { ParamSection } from "../ParamSection";
import { ScriptArgControl } from "./scripts/ScriptArgControl";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

export function ScriptsTab() {
  const { data: scriptsList } = useScriptsList();
  const { data: scriptInfo } = useScriptInfo();
  const store = useScriptStore();

  const selectableScripts = scriptsList?.txt2img ?? [];

  const selectedInfo = useMemo(
    () => scriptInfo?.find((s) => s.name === store.selectedScript),
    [scriptInfo, store.selectedScript],
  );

  const alwaysOnScripts = useMemo(
    () => scriptInfo?.filter((s) => s.is_alwayson && s.args.length > 0) ?? [],
    [scriptInfo],
  );

  return (
    <div className="flex flex-col gap-3 text-sm">
      <ParamSection title="Script">
        <div className="flex items-center gap-2">
          <Label className="text-[11px] text-muted-foreground w-16 flex-shrink-0">Script</Label>
          <Select value={store.selectedScript || "__none__"} onValueChange={(v) => store.setSelectedScript(v === "__none__" ? "" : v)}>
            <SelectTrigger className="h-7 text-xs flex-1">
              <SelectValue placeholder="None" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="__none__">None</SelectItem>
              {selectableScripts.map((name) => (
                <SelectItem key={name} value={name}>{name}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {selectedInfo && selectedInfo.args.length > 0 && (
          <div className="flex flex-col gap-2 mt-1">
            {selectedInfo.args.map((arg, i) => (
              <ScriptArgControl
                key={`${store.selectedScript}-${i}`}
                arg={arg}
                value={store.scriptArgs[i] ?? arg.value}
                onChange={(v) => store.setScriptArg(i, v)}
              />
            ))}
          </div>
        )}
      </ParamSection>

      {alwaysOnScripts.length > 0 && (
        <ParamSection title="Always-on Scripts" defaultOpen={false}>
          {alwaysOnScripts.map((script) => (
            <div key={script.name} className="flex flex-col gap-2 mb-3">
              <Label className="text-[11px] font-medium text-foreground">{script.name}</Label>
              {script.args.map((arg, i) => (
                <ScriptArgControl
                  key={`${script.name}-${i}`}
                  arg={arg}
                  value={store.alwaysOnOverrides[script.name]?.[i] ?? arg.value}
                  onChange={(v) => store.setAlwaysOnArg(script.name, i, v)}
                />
              ))}
            </div>
          ))}
        </ParamSection>
      )}
    </div>
  );
}
