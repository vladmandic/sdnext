import type { ComboboxGroup } from "@/components/ui/combobox";
import type { XyzAxisOption } from "@/api/hooks/useXyzAxisOptions";

export interface XyzAxisConfig {
  type: string;
  values: string;
}

export interface XyzGridConfig {
  x: XyzAxisConfig;
  y: XyzAxisConfig;
  z: XyzAxisConfig;
  drawLegend: boolean;
  includeGrid: boolean;
  includeSubgrids: boolean;
  includeImages: boolean;
  includeTime: boolean;
  includeText: boolean;
  marginSize: number;
}

/**
 * Maps a structured config to the positional script_args array expected by the XYZ Grid script's run() method.
 * The axis type values are integer indices into the full axis_options list.
 */
export function buildXyzScriptArgs(config: XyzGridConfig, axisOptions: XyzAxisOption[]): unknown[] {
  const findIndex = (label: string) => {
    if (!label) return 0; // "Nothing" is always index 0
    const idx = axisOptions.findIndex((o) => o.label === label);
    return idx >= 0 ? idx : 0;
  };

  return [
    findIndex(config.x.type), config.x.values, "",       // 0-2: x_type (index), x_values, x_values_dropdown
    findIndex(config.y.type), config.y.values, "",       // 3-5: y_type, y_values, y_values_dropdown
    findIndex(config.z.type), config.z.values, "",       // 6-8: z_type, z_values, z_values_dropdown
    false,                                                // 9: csv_mode
    config.drawLegend,                                    // 10: draw_legend
    false,                                                // 11: no_fixed_seeds
    config.includeGrid,                                   // 12: include_grid
    config.includeSubgrids,                               // 13: include_subgrids
    config.includeImages,                                 // 14: include_images
    config.includeTime,                                   // 15: include_time
    config.includeText,                                   // 16: include_text
    config.marginSize,                                    // 17: margin_size
    false,                                                // 18: create_video
    "None",                                               // 19: video_type
    2.0,                                                  // 20: video_duration
    false,                                                // 21: video_loop
    0,                                                    // 22: video_pad
    0,                                                    // 23: video_interpolate
  ];
}

const RANGE_RE = /^(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)\s*:\s*(\d+)$/;

/** Parses a values string and returns the count of discrete values. */
export function countAxisValues(values: string, type: string): number {
  const trimmed = values.trim();
  if (!trimmed) return 0;

  const parts = trimmed.split(",").map((s) => s.trim()).filter(Boolean);
  if (parts.length === 0) return 0;

  if (type === "int" || type === "float") {
    let total = 0;
    for (const part of parts) {
      const m = RANGE_RE.exec(part);
      if (m) {
        total += Math.max(2, parseInt(m[3], 10));
      } else {
        total += 1;
      }
    }
    return total;
  }

  return parts.length;
}

export const AXIS_GROUPS: Array<{ heading: string; match: string }> = [
  { heading: "Parameters", match: "[Param]" },
  { heading: "Sampler", match: "[Sampler]" },
  { heading: "Guidance", match: "[Guidance]" },
  { heading: "Model", match: "[Model]" },
  { heading: "Network", match: "[Network]" },
  { heading: "Prompt", match: "[Prompt]" },
  { heading: "Refine / Hires", match: "[Refine]" },
  { heading: "Postprocess", match: "[Postprocess]" },
  { heading: "Quantization", match: "[Quant]" },
  { heading: "Advanced", match: "" },
];

/** Groups the flat axis options list into ComboboxGroup[] for the axis type picker. */
export function groupAxisOptions(options: XyzAxisOption[]): ComboboxGroup[] {
  const groups: ComboboxGroup[] = AXIS_GROUPS.map((g) => ({ heading: g.heading, options: [] as string[] }));
  const advancedGroup = groups[groups.length - 1];

  for (const opt of options) {
    if (opt.label === "Nothing") continue;
    let placed = false;
    for (let i = 0; i < AXIS_GROUPS.length - 1; i++) {
      if (opt.label.startsWith(AXIS_GROUPS[i].match)) {
        (groups[i].options as string[]).push(opt.label);
        placed = true;
        break;
      }
    }
    if (!placed) {
      (advancedGroup.options as string[]).push(opt.label);
    }
  }

  return groups.filter((g) => g.options.length > 0);
}
