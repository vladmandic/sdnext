export interface ArgGroup {
  title?: string;
  args: number[];
  grid?: boolean;
  disabledUnless?: number;
  argDisabledUnless?: Record<number, number>;  // Per-arg disable: arg index → controller arg index (truthiness check)
  argRequires?: Record<number, { arg: number; includes: string }>;  // Per-arg disable: arg disabled unless another arg's value includes a substring
}

export interface ScriptLayout {
  hidden?: number[];
  headerRow?: number[];  // Arg indices rendered alongside an Enabled toggle in a top row
  headerArgRequires?: Record<number, { arg: number; includes: string }>;  // Header arg disabled unless another arg's value includes a substring
  groups: ArgGroup[];
}

/**
 * Per-script layout descriptors.
 * Keys are lowercase script names (matching ScriptInfo.name).
 * Unknown scripts fall back to the flat arg list.
 */
export const SCRIPT_LAYOUTS: Record<string, ScriptLayout> = {
  nudenet: {
    groups: [
      { args: [0] },
      { title: "Detection", args: [6, 7], disabledUnless: 0 },
      { title: "Censoring", args: [8, 9, 10, 5], disabledUnless: 0 },
      { title: "Policy", args: [4, 1, 2, 3, 11, 12, 13], grid: true, disabledUnless: 0 },
    ],
  },

  "prompt enhance": {
    hidden: [0, 1],
    headerRow: [2, 15],
    headerArgRequires: { 15: { arg: 3, includes: "\uf06e" } },  // Use vision disabled unless LLM model has vision symbol
    groups: [
      { args: [3] },
      { title: "Prompts", args: [4, 5, 6] },
      { title: "Options", args: [13, 18, 14, 17, 16], grid: true, argDisabledUnless: { 17: 16, 18: 13 }, argRequires: { 13: { arg: 3, includes: "\uf0eb" } } },
      { title: "Sampling", args: [8, 7, 9, 10, 11, 12], disabledUnless: 8 },
    ],
  },

  "xyz grid": {
    hidden: [3, 6, 9, 19, 20, 21, 22, 23, 24],
    groups: [
      { args: [0] },
      { title: "Axes", args: [1, 2, 4, 5, 7, 8, 10], disabledUnless: 0 },
      { title: "Options", args: [11, 12, 13, 14, 15, 16, 17, 18], grid: true, disabledUnless: 0 },
    ],
  },
};
