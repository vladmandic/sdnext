import { Switch } from "@/components/ui/switch";
import { useScriptStore } from "@/stores/scriptStore";
import { ScriptArgControl } from "./ScriptArgControl";
import { SCRIPT_LAYOUTS } from "./scriptLayouts";
import type { ScriptInfoV2 } from "@/api/types/script";

interface ScriptSectionProps {
  script: ScriptInfoV2;
  getArgValue: (index: number) => unknown;
  setArgValue: (index: number, value: unknown) => void;
}

export function ScriptSection({ script, getArgValue, setArgValue }: ScriptSectionProps) {
  const layout = SCRIPT_LAYOUTS[script.name];
  const sectionEnabled = useScriptStore((s) => s.alwaysOnEnabled[script.name] ?? false);
  const setSectionEnabled = useScriptStore((s) => s.setAlwaysOnEnabled);

  // Fallback: flat list for unknown scripts
  if (!layout) {
    return (
      <div className="flex flex-col gap-2">
        {script.args.map((arg, i) => (
          <ScriptArgControl key={i} arg={arg} value={getArgValue(i)} onChange={(v) => setArgValue(i, v)} />
        ))}
      </div>
    );
  }

  const hiddenSet = new Set(layout.hidden ?? []);
  const hasHeaderRow = layout.headerRow && layout.headerRow.length > 0;

  return (
    <div className="flex flex-col gap-3">
      {hasHeaderRow && (
        <div className="flex items-center gap-4">
          <label className="flex items-center gap-1.5 text-2xs text-muted-foreground cursor-pointer">
            <Switch size="sm" checked={sectionEnabled} onCheckedChange={(checked) => setSectionEnabled(script.name, checked)} />
            Enabled
          </label>
          {layout.headerRow!.map((idx) => {
            const arg = script.args[idx];
            if (!arg || typeof (getArgValue(idx) ?? arg.value) !== "boolean") return null;
            const requirement = layout.headerArgRequires?.[idx];
            const requirementMet = !requirement || String(getArgValue(requirement.arg) ?? "").includes(requirement.includes);
            const disabled = !sectionEnabled || !requirementMet;
            return (
              <label key={idx} className={`flex items-center gap-1.5 text-2xs text-muted-foreground cursor-pointer${disabled ? " opacity-50 pointer-events-none" : ""}`}>
                <Switch size="sm" checked={Boolean(getArgValue(idx) ?? arg.value)} onCheckedChange={(checked) => setArgValue(idx, checked)} disabled={disabled} />
                {arg.label}
              </label>
            );
          })}
        </div>
      )}

      {layout.groups.map((group, gi) => {
        const sectionDisabled = hasHeaderRow && !sectionEnabled;
        const ownDisabled = group.disabledUnless != null ? !getArgValue(group.disabledUnless) : false;
        const visibleArgs = group.args.filter((idx) => !hiddenSet.has(idx) && idx < script.args.length);
        if (visibleArgs.length === 0) return null;

        // Split into booleans and non-booleans for grid rendering
        const boolArgs = group.grid ? visibleArgs.filter((idx) => typeof (getArgValue(idx) ?? script.args[idx]?.value) === "boolean") : [];
        const otherArgs = group.grid ? visibleArgs.filter((idx) => typeof (getArgValue(idx) ?? script.args[idx]?.value) !== "boolean") : visibleArgs;

        return (
          <div key={gi} className="flex flex-col gap-2">
            {group.title && (
              <span className={`text-3xs font-medium uppercase tracking-wider text-muted-foreground/70${sectionDisabled || ownDisabled ? " opacity-50" : ""}`}>{group.title}</span>
            )}

            {group.grid && boolArgs.length > 0 && (
              <div className="grid grid-cols-2 gap-2">
                {boolArgs.map((idx) => {
                  const arg = script.args[idx];
                  const val = getArgValue(idx);
                  // Controller is exempt from its own group condition, but not from section disable
                  const isController = idx === group.disabledUnless;
                  const perArgController = group.argDisabledUnless?.[idx];
                  const requirement = group.argRequires?.[idx];
                  const requirementMet = !requirement || String(getArgValue(requirement.arg) ?? "").includes(requirement.includes);
                  const disabled = sectionDisabled || (ownDisabled && !isController) || (perArgController != null && !getArgValue(perArgController)) || !requirementMet;
                  return (
                    <label key={idx} className={`flex items-center gap-1.5 text-2xs text-muted-foreground cursor-pointer${disabled ? " opacity-50 pointer-events-none" : ""}`}>
                      <Switch size="sm" checked={Boolean(val ?? arg.value)} onCheckedChange={(checked) => setArgValue(idx, checked)} disabled={disabled} />
                      {arg.label}
                    </label>
                  );
                })}
              </div>
            )}

            {otherArgs.map((idx) => {
              const arg = script.args[idx];
              // Controller is exempt from its own group condition, but not from section disable
              const isController = idx === group.disabledUnless;
              const perArgController = group.argDisabledUnless?.[idx];
              const requirement = group.argRequires?.[idx];
              const requirementMet = !requirement || String(getArgValue(requirement.arg) ?? "").includes(requirement.includes);
              const argDisabled = sectionDisabled || (ownDisabled && !isController) || (perArgController != null && !getArgValue(perArgController)) || !requirementMet;
              return (
                <ScriptArgControl key={idx} arg={arg} value={getArgValue(idx)} onChange={(v) => setArgValue(idx, v)} disabled={argDisabled} />
              );
            })}
          </div>
        );
      })}
    </div>
  );
}
