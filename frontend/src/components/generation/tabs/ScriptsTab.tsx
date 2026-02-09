import { useCallback, useMemo } from "react";
import { useScriptsList, useScriptInfo } from "@/api/hooks/useScripts";
import { useScriptStore } from "@/stores/scriptStore";
import { ParamSection } from "../ParamSection";
import { ScriptArgControl } from "./scripts/ScriptArgControl";
import { ScriptSection } from "./scripts/ScriptSection";
import { Label } from "@/components/ui/label";
import { Combobox } from "@/components/ui/combobox";
import type { ScriptInfo } from "@/api/types/script";

/** Scripts handled by dedicated tabs (e.g. IP Adapters lives in Input tab). */
const HIDDEN_ALWAYS_ON = new Set(["ip adapters"]);

export function ScriptsTab() {
  const { data: scriptsList } = useScriptsList();
  const { data: scriptInfo } = useScriptInfo();
  const store = useScriptStore();

  const selectableScripts = scriptsList?.txt2img ?? [];

  const selectedInfo = useMemo(
    () => scriptInfo?.find((s) => s.name === store.selectedScript),
    [scriptInfo, store.selectedScript],
  );

  // Deduplicate by name (API returns one per context: txt2img, img2img, control)
  // and exclude scripts that have dedicated tabs.
  const alwaysOnScripts = useMemo(() => {
    if (!scriptInfo) return [];
    const seen = new Set<string>();
    return scriptInfo.filter((s) => {
      if (!s.is_alwayson || s.args.length === 0) return false;
      if (HIDDEN_ALWAYS_ON.has(s.name)) return false;
      if (seen.has(s.name)) return false;
      seen.add(s.name);
      return true;
    });
  }, [scriptInfo]);

  return (
    <div className="flex flex-col gap-3 text-sm">
      <ParamSection title="Script">
        <div className="flex items-center gap-2">
          <Label className="text-[11px] text-muted-foreground w-16 flex-shrink-0">Script</Label>
          <Combobox
            value={store.selectedScript || "__none__"}
            onValueChange={(v) => store.setSelectedScript(v === "__none__" ? "" : v)}
            options={[
              { value: "__none__", label: "None" },
              ...selectableScripts.map((name) => ({ value: name, label: name })),
            ]}
            placeholder="None"
            className="h-7 text-xs flex-1"
          />
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

      {alwaysOnScripts.map((script) => (
        <AlwaysOnSection key={script.name} script={script} />
      ))}
    </div>
  );
}

function AlwaysOnSection({ script }: { script: ScriptInfo }) {
  const store = useScriptStore();
  const getArgValue = useCallback(
    (index: number) => store.alwaysOnOverrides[script.name]?.[index] ?? script.args[index]?.value,
    [store.alwaysOnOverrides, script.name, script.args],
  );
  const setArgValue = useCallback(
    (index: number, value: unknown) => store.setAlwaysOnArg(script.name, index, value),
    [store, script.name],
  );

  return (
    <ParamSection title={script.name} defaultOpen={false}>
      <ScriptSection script={script} getArgValue={getArgValue} setArgValue={setArgValue} />
    </ParamSection>
  );
}
