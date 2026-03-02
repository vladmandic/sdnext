import { useMemo } from "react";
import { useCanvasStore } from "@/stores/canvasStore";
import { useControlStore } from "@/stores/controlStore";
import { UNIT_TYPE_LABELS, EXCLUSIVE_CONTROL_TYPES } from "@/api/types/control";
import type { InputRole, ControlUnitType } from "@/api/types/control";

export interface UnifiedInput {
  index: number;
  role: InputRole;
  unitType?: ControlUnitType;
  source: { type: "canvas" } | { type: "unit"; unitIndex: number };
  enabled: boolean;
  label: string;
}

export interface UnifiedInputsResult {
  inputs: UnifiedInput[];
  lockedControlType: ControlUnitType | null;
  availableControlSubTypes: { value: ControlUnitType; label: string; disabled: boolean }[];
  toUnitIndex: (unifiedIndex: number) => number;
  toUnifiedIndex: (unitIndex: number) => number;
}

export function useUnifiedInputs(): UnifiedInputsResult {
  const inputRole = useCanvasStore((s) => s.inputRole);
  const units = useControlStore((s) => s.units);

  return useMemo(() => {
    const inputs: UnifiedInput[] = [];

    // Input #1: canvas input frame
    const canvasRole: InputRole = inputRole === "reference" ? "reference" : "initial";
    inputs.push({
      index: 1,
      role: canvasRole,
      source: { type: "canvas" },
      enabled: true,
      label: `Input 1 (${canvasRole === "initial" ? "Initial" : "Reference"})`,
    });

    // Input #2+: control units
    for (let i = 0; i < units.length; i++) {
      const u = units[i];
      const idx = i + 2;
      const role: InputRole = u.unitType === "reference" ? "reference" : "control";
      const roleLabel = role === "reference"
        ? "Reference"
        : `Control: ${UNIT_TYPE_LABELS[u.unitType]}`;
      inputs.push({
        index: idx,
        role,
        unitType: u.unitType,
        source: { type: "unit", unitIndex: i },
        enabled: u.enabled,
        label: `Input ${idx} (${roleLabel})`,
      });
    }

    // Locked exclusive control type
    const lockedControlType = units
      .filter((u) => u.enabled && EXCLUSIVE_CONTROL_TYPES.has(u.unitType))
      .map((u) => u.unitType)[0] ?? null;

    // Available sub-types for "Add Input > Control"
    const allSubTypes: ControlUnitType[] = [
      "controlnet", "t2i", "xs", "lite", "style_transfer", "ip",
    ];
    const availableControlSubTypes = allSubTypes.map((t) => ({
      value: t,
      label: UNIT_TYPE_LABELS[t],
      disabled: lockedControlType !== null
        && EXCLUSIVE_CONTROL_TYPES.has(t)
        && t !== lockedControlType,
    }));

    return {
      inputs,
      lockedControlType,
      availableControlSubTypes,
      toUnitIndex: (ui: number) => (ui === 1 ? -1 : ui - 2),
      toUnifiedIndex: (ui: number) => (ui === -1 ? 1 : ui + 2),
    };
  }, [inputRole, units]);
}
