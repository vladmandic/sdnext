import { useCallback, useMemo } from "react";
import { useScripts } from "@/api/hooks/useScripts";
import { useScriptStore } from "@/stores/scriptStore";
import { ParamSection } from "../ParamSection";
import { ScriptArgControl } from "./scripts/ScriptArgControl";
import { ScriptSection } from "./scripts/ScriptSection";
import { Label } from "@/components/ui/label";
import { Combobox } from "@/components/ui/combobox";
import type { ScriptInfoV2 } from "@/api/types/script";

export function ScriptsTab() {
  const { data: scripts } = useScripts();
  const store = useScriptStore();

  const selectableScripts = useMemo(
    () => scripts?.scripts.filter((s) => !s.is_alwayson && s.contexts.includes("txt2img")).map((s) => s.name) ?? [],
    [scripts],
  );

  const selectedInfo = useMemo(
    () => scripts?.scripts.find((s) => s.name === store.selectedScript),
    [scripts, store.selectedScript],
  );

  const alwaysOnScripts = useMemo(() => {
    if (!scripts) return [];
    return scripts.scripts.filter((s) => s.is_alwayson && s.args.length > 0);
  }, [scripts]);

  return (
    <div className="flex flex-col gap-3 text-sm">
      <ParamSection title="Script">
        <div className="flex items-center gap-2">
          <Label className="text-2xs text-muted-foreground w-16 flex-shrink-0">Script</Label>
          <Combobox
            value={store.selectedScript || "__none__"}
            onValueChange={(v) => store.setSelectedScript(v === "__none__" ? "" : v)}
            options={[
              { value: "__none__", label: "None" },
              ...selectableScripts.map((name) => ({ value: name, label: name })),
            ]}
            placeholder="None"
            className="h-6 text-2xs flex-1"
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

function AlwaysOnSection({ script }: { script: ScriptInfoV2 }) {
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
