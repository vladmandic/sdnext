import { useState, useMemo, useEffect, useCallback, useRef } from "react";
import { toast } from "sonner";
import { useOptions, useSetOptions, useOptionsInfo } from "@/api/hooks/useSettings";
import { useModelList, useSamplerList, useVaeList, useUpscalerList } from "@/api/hooks/useModels";
import type { OptionInfoMeta } from "@/api/types/settings";
import type { SettingSectionDef, SettingDef } from "@/lib/settingsSchema";
import { settingsSchema, getSettingsMap, metaToSettingDef } from "@/lib/settingsSchema";
import { getParamHelpPlain } from "@/data/parameterHelp";
import { SettingsSection } from "./SettingsSection";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { cn } from "@/lib/utils";
import { Save, RotateCcw, Search, ListRestart, Plug, Unplug, Check } from "lucide-react";
import { useUiStore } from "@/stores/uiStore";
import type { CornerStyle, ColorMode } from "@/stores/uiStore";
import { useConnectionStore } from "@/stores/connectionStore";
import { api } from "@/api/client";
import { ws } from "@/api/wsManager";
import { queryClient } from "@/main";

const CONNECTION_SECTION_ID = "__connection";
const APPEARANCE_SECTION_ID = "__appearance";

const CORNER_STYLES: { value: CornerStyle; label: string }[] = [
  { value: "rounded", label: "Rounded" },
  { value: "square", label: "Square" },
];

const COLOR_MODES: { value: ColorMode; label: string }[] = [
  { value: "dark", label: "Dark" },
  { value: "light", label: "Light" },
  { value: "system", label: "System" },
];

/** Backend settings that are Gradio-specific and meaningless to the React UI */
const GRADIO_ONLY_KEYS = new Set([
  // Model selector — accessible from the toolbar
  "sd_model_checkpoint",
  // UI section — Gradio/ModernUI layout and appearance
  "theme_type", "theme_style", "gradio_theme", "quicksettings_list",
  "ui_request_timeout", "ui_disabled", "compact_view", "ui_columns",
  "logmonitor_show", "logmonitor_refresh_period", "send_seed", "send_size",
  "font_size", "ui_locale",
  "extra_networks_card_size", "extra_networks_card_cover", "extra_networks_card_square",
  "autolaunch", "motd", "subpath", "gpu_monitor",
  // ModernUI extension settings
  "uiux_grid_image_size", "uiux_panel_min_width", "uiux_hide_legacy",
  "uiux_persist_layout", "uiux_no_slider_layout",
  "uiux_show_labels_aside", "uiux_show_labels_main", "uiux_show_labels_tabs",
  "uiux_show_input_range_ticks", "uiux_no_headers_params", "uiux_show_outline_params",
  "uiux_default_layout", "uiux_mobile_scale",
  // Extra networks section
  "extra_networks_show", "extra_networks_view",
  "extra_networks_sidebar_width", "extra_networks_height", "extra_networks_fetch",
  // Live preview section
  "live_preview_refresh_period", "notification_audio_enable", "notification_audio_path",
  // Inpaint settings — per-request in generation UI
  "img2img_color_correction", "color_correction_method",
  "mask_apply_overlay", "inpainting_mask_weight",
  "img2img_background_color", "initial_noise_multiplier",
  // Aspect ratios — will be consumed by generation panel directly
  "aspect_ratios",
]);

/** Remap settings from one backend section to another (e.g. out of the now-empty "User Interface" section) */
const SECTION_REMAP: Record<string, string> = {
  return_grid: "saving-images",
  return_mask: "saving-images",
  return_mask_composite: "saving-images",
};

/** Sentinel strings matching backend defaults in modules/shared.py — used for model/VAE "no selection" states */
const SENTINEL_NONE = "None";
const SENTINEL_AUTOMATIC = "Automatic";

const HEX_REGEX = /^#[0-9a-fA-F]{6}$/;

/** Build a SettingDef: backend metadata is authoritative for choices, curated provides label/description/component overrides */
function buildSettingDef(
  key: string,
  info: OptionInfoMeta,
  curatedMap: Map<string, { section: SettingSectionDef; setting: SettingDef }>,
): SettingDef {
  const curated = curatedMap.get(key);
  if (!curated) return metaToSettingDef(key, info);
  const base = metaToSettingDef(key, info);
  const label = curated.setting.label;
  return {
    ...base,
    label,
    description: getParamHelpPlain(label) || base.description,
    component: curated.setting.component,
    ...(curated.setting.min !== undefined && { min: curated.setting.min }),
    ...(curated.setting.max !== undefined && { max: curated.setting.max }),
    ...(curated.setting.step !== undefined && { step: curated.setting.step }),
    ...(curated.setting.defaultValue !== undefined && { defaultValue: curated.setting.defaultValue }),
    ...(curated.setting.requiresRestart && { requiresRestart: true }),
    ...(curated.setting.baseFolderKey && { baseFolderKey: curated.setting.baseFolderKey }),
    // choices: prefer backend (base), fall back to curated if backend has none
    choices: base.choices ?? curated.setting.choices,
  };
}

function SettingRow({ label, description, inline, children }: { label: string; description?: string; inline?: boolean; children: React.ReactNode }) {
  const labelBlock = (
    <div className="flex flex-col gap-0.5">
      <span className="text-xs font-medium">{label}</span>
      {description && <span className="text-3xs text-muted-foreground leading-tight">{description}</span>}
    </div>
  );
  if (inline) {
    return (
      <div className="flex items-center justify-between gap-3">
        {labelBlock}
        {children}
      </div>
    );
  }
  return (
    <div className="flex flex-col gap-1">
      {labelBlock}
      {children}
    </div>
  );
}

function SegmentedControl<T extends string>({ options, value, onChange }: { options: { value: T; label: string }[]; value: T; onChange: (v: T) => void }) {
  return (
    <div className="inline-flex border border-border bg-muted/40 p-0.5" style={{ borderRadius: "var(--control-radius)" }}>
      {options.map((opt) => (
        <button
          key={opt.value}
          type="button"
          onClick={() => onChange(opt.value)}
          className={cn(
            "px-2.5 py-1 text-xs transition-colors",
            value === opt.value
              ? "bg-background text-foreground shadow-sm"
              : "text-muted-foreground hover:text-foreground",
          )}
          style={{ borderRadius: "var(--control-inner-radius)" }}
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
}

function AppearancePanel() {
  const colorMode = useUiStore((s) => s.colorMode);
  const setColorMode = useUiStore((s) => s.setColorMode);
  const accentColor = useUiStore((s) => s.accentColor);
  const setAccentColor = useUiStore((s) => s.setAccentColor);
  const cornerStyle = useUiStore((s) => s.cornerStyle);
  const setCornerStyle = useUiStore((s) => s.setCornerStyle);
  const borderRadius = useUiStore((s) => s.borderRadius);
  const setBorderRadius = useUiStore((s) => s.setBorderRadius);
  const uiScale = useUiStore((s) => s.uiScale);
  const setUiScale = useUiStore((s) => s.setUiScale);
  const canvasLabelScale = useUiStore((s) => s.canvasLabelScale);
  const setCanvasLabelScale = useUiStore((s) => s.setCanvasLabelScale);

  const [hexInput, setHexInput] = useState(accentColor);
  useEffect(() => { setHexInput(accentColor); }, [accentColor]);

  return (
    <div>
      <h3 className="text-sm font-medium mb-4">Appearance</h3>
      <div className="space-y-4">
        <SettingRow label="Color mode" description="Overall color scheme of the interface">
          <SegmentedControl options={COLOR_MODES} value={colorMode} onChange={setColorMode} />
        </SettingRow>

        <SettingRow label="Accent color" description="Primary color used for buttons, links, and highlights" inline>
          <div className="flex items-center gap-2">
            <input
              type="color"
              value={accentColor}
              onChange={(e) => setAccentColor(e.target.value)}
              className="h-7 w-7 cursor-pointer border border-border rounded-sm bg-transparent p-0"
            />
            <Input
              value={hexInput}
              onChange={(e) => {
                setHexInput(e.target.value);
                if (HEX_REGEX.test(e.target.value)) setAccentColor(e.target.value);
              }}
              className="h-7 w-20 text-xs font-mono"
            />
          </div>
        </SettingRow>

        <SettingRow label="Corner style" description="Shape of toggle switches and segmented controls">
          <SegmentedControl options={CORNER_STYLES} value={cornerStyle} onChange={setCornerStyle} />
        </SettingRow>

        <SettingRow label="Border radius" description="Roundness of cards, inputs, and panels">
          <div className="flex items-center gap-2 flex-1">
            <Slider
              min={0}
              max={1}
              step={0.05}
              value={[borderRadius]}
              onValueChange={([v]) => setBorderRadius(v)}
              className="flex-1"
            />
            <span className="text-xs text-muted-foreground tabular-nums w-14 text-right">{borderRadius.toFixed(2)}rem</span>
          </div>
        </SettingRow>

        <SettingRow label="UI scale" description="Base font size — all spacing and controls scale proportionally">
          <div className="flex items-center gap-2 flex-1">
            <Slider
              min={8}
              max={28}
              step={1}
              value={[uiScale]}
              onValueChange={([v]) => setUiScale(v)}
              className="flex-1"
            />
            <span className="text-xs text-muted-foreground tabular-nums w-14 text-right">{uiScale}px</span>
          </div>
        </SettingRow>

        <SettingRow label="Canvas label scale" description="Size of frame labels and floating control panels on the canvas">
          <div className="flex items-center gap-2 flex-1">
            <Slider
              min={0.5}
              max={2}
              step={0.1}
              value={[canvasLabelScale]}
              onValueChange={([v]) => setCanvasLabelScale(v)}
              className="flex-1"
            />
            <span className="text-xs text-muted-foreground tabular-nums w-14 text-right">{canvasLabelScale.toFixed(1)}x</span>
          </div>
        </SettingRow>
      </div>
    </div>
  );
}

function maskValue(v: string) {
  if (v.length <= 4) return "\u2022".repeat(v.length);
  return `${v.slice(0, 2)}${"•".repeat(Math.min(v.length - 4, 6))}${v.slice(-2)}`;
}

function ConnectionPanel() {
  const backendUrl = useConnectionStore((s) => s.backendUrl);
  const username = useConnectionStore((s) => s.username);
  const password = useConnectionStore((s) => s.password);
  const storeSetUrl = useConnectionStore((s) => s.setBackendUrl);
  const storeSetAuth = useConnectionStore((s) => s.setAuth);
  const storeReset = useConnectionStore((s) => s.reset);

  const [urlInput, setUrlInput] = useState(backendUrl);
  const [status, setStatus] = useState<"idle" | "checking" | "connected" | "unreachable">("idle");

  const authConfigured = username.length > 0 || password.length > 0;
  const [editingAuth, setEditingAuth] = useState(false);
  const [userDraft, setUserDraft] = useState("");
  const [passDraft, setPassDraft] = useState("");

  useEffect(() => { setUrlInput(backendUrl); }, [backendUrl]);

  const checkConnection = useCallback(async (baseUrl: string) => {
    setStatus("checking");
    try {
      await fetch(`${baseUrl}/sdapi/v2/server-info`, { signal: AbortSignal.timeout(5000) });
      setStatus("connected");
    } catch {
      setStatus("unreachable");
    }
  }, []);

  useEffect(() => {
    if (backendUrl) {
      checkConnection(backendUrl);
    } else {
      setStatus("connected");
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const saveAuth = useCallback((user: string, pass: string) => {
    storeSetAuth(user, pass);
    if (user && pass) api.setAuth(user, pass);
    else api.clearAuth();
  }, [storeSetAuth]);

  const handleConnect = useCallback(async () => {
    const effectiveUrl = urlInput.replace(/\/$/, "") || window.location.origin;
    api.setBaseUrl(effectiveUrl);
    storeSetUrl(urlInput.replace(/\/$/, ""));
    ws.updateUrl(api.getWebSocketUrl("/sdapi/v2/ws"));
    queryClient.invalidateQueries();
    await checkConnection(effectiveUrl);
    toast.success("Connection updated");
  }, [urlInput, storeSetUrl, checkConnection]);

  const handleReset = useCallback(() => {
    storeReset();
    api.setBaseUrl(window.location.origin);
    api.clearAuth();
    ws.updateUrl(api.getWebSocketUrl("/sdapi/v2/ws"));
    queryClient.invalidateQueries();
    setStatus("connected");
    setEditingAuth(false);
    toast.success("Connection reset to default");
  }, [storeReset]);

  return (
    <div>
      <h3 className="text-sm font-medium mb-4">Connection</h3>
      <div className="space-y-4">
        <SettingRow label="Backend URL" description="Leave empty to use the current origin">
          <div className="flex items-center gap-2">
            <Input
              value={urlInput}
              onChange={(e) => setUrlInput(e.target.value)}
              placeholder={window.location.origin}
              className="h-7 text-xs font-mono flex-1"
            />
            {status === "checking" && <span className="text-xs text-muted-foreground">Checking...</span>}
            {status === "connected" && <span className="text-xs text-emerald-500">Connected</span>}
            {status === "unreachable" && <span className="text-xs text-destructive">Unreachable</span>}
          </div>
        </SettingRow>

        <SettingRow label="Credentials" description="Basic auth username and password (optional)">
          {authConfigured && !editingAuth ? (
            <div className="flex items-center gap-2">
              <Check className="h-3 w-3 text-green-500 shrink-0" />
              <span className="text-xs text-muted-foreground font-mono">{maskValue(username)}</span>
              <span className="text-xs text-muted-foreground">/</span>
              <span className="text-xs text-muted-foreground font-mono">{maskValue(password)}</span>
              <Button size="xs" variant="ghost" onClick={() => { setEditingAuth(true); setUserDraft(""); setPassDraft(""); }}>Change</Button>
              <Button size="xs" variant="ghost" className="text-destructive" onClick={() => saveAuth("", "")}>Remove</Button>
            </div>
          ) : (
            <div className="flex items-center gap-2">
              <Input
                value={userDraft}
                onChange={(e) => setUserDraft(e.target.value)}
                placeholder="Username"
                autoComplete="off"
                className="h-6 text-2xs flex-1"
              />
              <Input
                value={passDraft}
                onChange={(e) => setPassDraft(e.target.value)}
                placeholder="Password"
                autoComplete="off"
                className="h-6 text-2xs flex-1"
              />
              <Button
                size="xs"
                disabled={!userDraft.trim() || !passDraft.trim()}
                onClick={() => { saveAuth(userDraft.trim(), passDraft.trim()); setEditingAuth(false); setUserDraft(""); setPassDraft(""); }}
              >
                Save
              </Button>
              {editingAuth && <Button size="xs" variant="ghost" onClick={() => { setEditingAuth(false); setUserDraft(""); setPassDraft(""); }}>Cancel</Button>}
            </div>
          )}
        </SettingRow>

        <div className="flex gap-2 pt-2">
          <Button size="sm" onClick={handleConnect} className="text-xs">
            <Plug size={14} />
            Connect
          </Button>
          <Button variant="secondary" size="sm" onClick={handleReset} className="text-xs">
            <Unplug size={14} />
            Reset to default
          </Button>
        </div>
      </div>
    </div>
  );
}

interface SettingsViewProps {
  onDirtyChange?: (isDirty: boolean) => void;
}

export function SettingsView({ onDirtyChange }: SettingsViewProps = {}) {
  const { data: options, isLoading } = useOptions();
  const setOptions = useSetOptions();
  const { data: optionsInfo, isLoading: isInfoLoading } = useOptionsInfo();
  const { data: models } = useModelList();
  const { data: samplers } = useSamplerList();
  const { data: vaes } = useVaeList();
  const { data: upscalers } = useUpscalerList();

  const [dirty, setDirty] = useState<Record<string, unknown>>({});
  const [activeSection, setActiveSection] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const searchInputRef = useRef<HTMLInputElement>(null);

  const curatedMap = useMemo(() => getSettingsMap(), []);

  const dynamicChoices = useMemo(() => {
    const choices: Record<string, string[]> = {};
    if (models) choices["sd_model_checkpoint"] = models.map((m) => m.title);
    if (models) choices["sd_model_refiner"] = [SENTINEL_NONE, ...models.map((m) => m.title)];
    if (vaes) choices["sd_vae"] = [SENTINEL_AUTOMATIC, SENTINEL_NONE, ...vaes.map((v) => v.name)];
    if (samplers) choices["sampler_name"] = samplers.map((s) => s.name);
    if (upscalers) choices["upscaler_for_img2img"] = upscalers.map((u) => u.name);
    return choices;
  }, [models, samplers, vaes, upscalers]);

  // Build all sections from backend metadata, with curated overrides
  const allSections = useMemo((): SettingSectionDef[] => {
    if (!optionsInfo || !options) return settingsSchema; // fallback before metadata loads

    const meta = optionsInfo.options;
    const result: SettingSectionDef[] = [];

    for (const section of optionsInfo.sections) {
      if (section.hidden) continue;

      const settings: SettingDef[] = [];
      let pendingSeparator: SettingDef | null = null;
      for (const [key, info] of Object.entries(meta)) {
        if ((SECTION_REMAP[key] ?? info.section_id) !== section.id) continue;
        if (!info.visible || info.hidden || info.is_legacy) continue;
        if (GRADIO_ONLY_KEYS.has(key)) continue;
        if (info.component === "separator") {
          if (info.label) pendingSeparator = { key, label: info.label, component: "separator" };
          continue;
        }
        if (!(key in options)) continue;

        if (pendingSeparator) { settings.push(pendingSeparator); pendingSeparator = null; }
        settings.push(buildSettingDef(key, info, curatedMap));
      }

      if (settings.length > 0) {
        result.push({ id: section.id, title: section.title, settings });
      }
    }

    return result;
  }, [optionsInfo, options, curatedMap]);

  // Resolve active section: use first section if none selected or selection is stale
  const resolvedActive = useMemo(() => {
    if (activeSection === CONNECTION_SECTION_ID) return CONNECTION_SECTION_ID;
    if (activeSection === APPEARANCE_SECTION_ID) return APPEARANCE_SECTION_ID;
    if (activeSection && allSections.some((s) => s.id === activeSection)) return activeSection;
    if (allSections.length > 0) return allSections[0].id;
    // Backend hasn't loaded yet — default to Connection so it's reachable
    return CONNECTION_SECTION_ID;
  }, [allSections, activeSection]);

  // Filter sections by search
  const filteredSections = useMemo(() => {
    if (!searchQuery) return allSections;
    const q = searchQuery.toLowerCase();
    return allSections
      .map((section) => ({
        ...section,
        settings: section.settings.filter(
          (s) =>
            s.key.toLowerCase().includes(q) ||
            s.label.toLowerCase().includes(q) ||
            (s.description?.toLowerCase().includes(q) ?? false) ||
            section.title.toLowerCase().includes(q),
        ),
      }))
      .filter((s) => s.settings.length > 0);
  }, [allSections, searchQuery]);

  function handleSettingChange(key: string, value: unknown) {
    setDirty((prev) => ({ ...prev, [key]: value }));
  }

  async function handleApply() {
    if (Object.keys(dirty).length === 0) return;
    try {
      await setOptions.mutateAsync(dirty);
      toast.success("Settings saved", { description: `${Object.keys(dirty).length} setting(s) updated` });
      setDirty({});
    } catch (err) {
      toast.error("Failed to save settings", { description: err instanceof Error ? err.message : String(err) });
    }
  }

  function handleReset() {
    setDirty({});
  }

  function handleRestoreDefaults() {
    const section = allSections.find((s) => s.id === resolvedActive);
    if (!section) return;
    const defaults: Record<string, unknown> = {};
    for (const setting of section.settings) {
      if (setting.component === "separator") continue;
      if (setting.defaultValue !== undefined) {
        defaults[setting.key] = setting.defaultValue;
      }
    }
    if (Object.keys(defaults).length > 0) {
      setDirty((prev) => ({ ...prev, ...defaults }));
    }
  }

  const dirtyCount = Object.keys(dirty).length;

  // Set of section IDs that have matching settings during search
  const matchingSectionIds = useMemo(() => {
    if (!searchQuery) return null;
    return new Set(filteredSections.map((s) => s.id));
  }, [searchQuery, filteredSections]);

  // Notify parent of dirty state changes
  useEffect(() => {
    onDirtyChange?.(dirtyCount > 0);
  }, [dirtyCount, onDirtyChange]);

  // Warn before page refresh with unsaved changes
  useEffect(() => {
    if (dirtyCount === 0) return;
    const handler = (e: BeforeUnloadEvent) => { e.preventDefault(); };
    window.addEventListener("beforeunload", handler);
    return () => window.removeEventListener("beforeunload", handler);
  }, [dirtyCount]);

  // Watch for command palette "Search settings" action
  const pendingSearch = useUiStore((s) => s.pendingSettingsSearch);
  const clearPendingSearch = useUiStore((s) => s.setPendingSettingsSearch);
  useEffect(() => {
    if (pendingSearch === null) return;
    clearPendingSearch(null);
    requestAnimationFrame(() => {
      setSearchQuery(pendingSearch);
      searchInputRef.current?.focus();
    });
  }, [pendingSearch, clearPendingSearch]);

  const handleNavigateToSection = useCallback((id: string) => {
    setActiveSection(id);
    setSearchQuery("");
  }, []);

  const backendReady = !isLoading && !isInfoLoading && !!options && !!optionsInfo;

  return (
    <div className="flex h-full">
      {/* Section navigation */}
      <div className="w-48 border-r border-border flex-shrink-0 flex flex-col">
        <div className="p-2">
          <div className="relative">
            <Search size={14} className="absolute left-2 top-1/2 -translate-y-1/2 text-muted-foreground" />
            <Input
              ref={searchInputRef}
              placeholder="Search..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              disabled={!backendReady}
              className="h-6 text-2xs pl-7"
            />
          </div>
        </div>
        <ScrollArea className="flex-1">
          <div className="flex flex-col gap-0.5 p-1">
            {allSections.map((section) => {
              const isMatch = matchingSectionIds?.has(section.id);
              const matchCount = searchQuery ? filteredSections.find((s) => s.id === section.id)?.settings.length : undefined;
              return (
                <button
                  key={section.id}
                  onClick={() => {
                    setActiveSection(section.id);
                    setSearchQuery("");
                  }}
                  disabled={!backendReady}
                  className={cn(
                    "text-left text-xs px-2 py-1.5 rounded-md transition-colors flex items-center justify-between",
                    "hover:bg-accent hover:text-accent-foreground",
                    resolvedActive === section.id && !searchQuery && "bg-accent text-accent-foreground font-medium",
                    matchingSectionIds && isMatch && "text-primary font-medium",
                    matchingSectionIds && !isMatch && "opacity-30",
                    !backendReady && "opacity-30 pointer-events-none",
                  )}
                >
                  <span>{section.title}</span>
                  {matchCount != null && matchCount > 0 && (
                    <span className="text-4xs bg-primary/10 text-primary rounded-full px-1.5 min-w-[1.125rem] text-center">
                      {matchCount}
                    </span>
                  )}
                </button>
              );
            })}
            {/* React UI local-only settings */}
            <div className="mt-2 pt-2 border-t border-border flex flex-col gap-0.5">
              <button
                onClick={() => {
                  setActiveSection(CONNECTION_SECTION_ID);
                  setSearchQuery("");
                }}
                className={cn(
                  "w-full text-left text-xs px-2 py-1.5 rounded-md transition-colors",
                  "hover:bg-accent hover:text-accent-foreground",
                  resolvedActive === CONNECTION_SECTION_ID && !searchQuery && "bg-accent text-accent-foreground font-medium",
                  matchingSectionIds && "opacity-30",
                )}
              >
                Connection
              </button>
              <button
                onClick={() => {
                  setActiveSection(APPEARANCE_SECTION_ID);
                  setSearchQuery("");
                }}
                className={cn(
                  "w-full text-left text-xs px-2 py-1.5 rounded-md transition-colors",
                  "hover:bg-accent hover:text-accent-foreground",
                  resolvedActive === APPEARANCE_SECTION_ID && !searchQuery && "bg-accent text-accent-foreground font-medium",
                  matchingSectionIds && "opacity-30",
                )}
              >
                Appearance
              </button>
            </div>
          </div>
        </ScrollArea>

        {/* Apply / Reset */}
        <div className="p-2 border-t border-border flex flex-col gap-1.5">
          <Button
            size="sm"
            onClick={handleApply}
            disabled={dirtyCount === 0 || setOptions.isPending}
            className="w-full text-xs"
          >
            <Save size={14} />
            Apply{dirtyCount > 0 && ` (${dirtyCount})`}
          </Button>
          <Button
            variant="secondary"
            size="sm"
            onClick={handleReset}
            disabled={dirtyCount === 0}
            className="w-full text-xs"
          >
            <RotateCcw size={14} />
            Reset
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={handleRestoreDefaults}
            disabled={!resolvedActive || !!searchQuery}
            className="w-full text-xs"
          >
            <ListRestart size={14} />
            Restore defaults
          </Button>
        </div>
      </div>

      {/* Settings content */}
      <ScrollArea className="flex-1">
        <div className="p-4 max-w-2xl">
          {resolvedActive === CONNECTION_SECTION_ID ? (
            <ConnectionPanel />
          ) : resolvedActive === APPEARANCE_SECTION_ID ? (
            <AppearancePanel />
          ) : !backendReady ? (
            <div className="flex items-center justify-center h-32 text-muted-foreground text-sm">
              Loading settings...
            </div>
          ) : searchQuery ? (
            // Search results across all sections
            <div className="space-y-6">
              {filteredSections.map((section) => (
                <SettingsSection
                  key={section.id}
                  section={section}
                  values={options}
                  dirty={dirty}
                  onSettingChange={handleSettingChange}
                  dynamicChoices={dynamicChoices}
                  searchQuery={searchQuery}
                  onNavigateToSection={handleNavigateToSection}
                />
              ))}
              {filteredSections.length === 0 && (
                <p className="text-sm text-muted-foreground text-center py-8">No settings match your search</p>
              )}
            </div>
          ) : (
            // Single section view
            (() => {
              const section = allSections.find((s) => s.id === resolvedActive);
              if (!section) return null;
              return (
                <SettingsSection
                  section={section}
                  values={options}
                  dirty={dirty}
                  onSettingChange={handleSettingChange}
                  dynamicChoices={dynamicChoices}
                />
              );
            })()
          )}
        </div>
      </ScrollArea>
    </div>
  );
}
