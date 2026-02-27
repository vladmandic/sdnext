import { useState, useMemo, useEffect, useCallback } from "react";
import { toast } from "sonner";
import { useOptions, useSetOptions, useOptionsInfo } from "@/api/hooks/useSettings";
import { useModelList, useSamplerList, useVaeList, useUpscalerList } from "@/api/hooks/useModels";
import type { OptionInfoMeta } from "@/api/types/settings";
import type { SettingSectionDef, SettingDef } from "@/lib/settingsSchema";
import { settingsSchema, getSettingsMap, metaToSettingDef } from "@/lib/settingsSchema";
import { SettingsSection } from "./SettingsSection";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import { Save, RotateCcw, Search, ListRestart, Plug, Unplug } from "lucide-react";
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
  // UI section
  "theme_type", "theme_style", "gradio_theme", "quicksettings_list",
  "ui_request_timeout", "ui_disabled", "compact_view", "ui_columns",
  "logmonitor_show", "logmonitor_refresh_period", "send_seed", "send_size",
  "font_size", "ui_locale",
  "extra_networks_card_size", "extra_networks_card_cover", "extra_networks_card_square",
  // Extra networks section
  "extra_networks_show", "extra_networks_view",
  "extra_networks_sidebar_width", "extra_networks_height", "extra_networks_fetch",
  // Live preview section
  "live_preview_refresh_period", "notification_audio_enable", "notification_audio_path",
]);

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
  return {
    ...base,
    label: curated.setting.label,
    ...(curated.setting.description !== undefined && { description: curated.setting.description }),
    component: curated.setting.component,
    ...(curated.setting.min !== undefined && { min: curated.setting.min }),
    ...(curated.setting.max !== undefined && { max: curated.setting.max }),
    ...(curated.setting.step !== undefined && { step: curated.setting.step }),
    ...(curated.setting.defaultValue !== undefined && { defaultValue: curated.setting.defaultValue }),
    ...(curated.setting.requiresRestart && { requiresRestart: true }),
    // choices: prefer backend (base), fall back to curated if backend has none
    choices: base.choices ?? curated.setting.choices,
  };
}

function AppearanceRow({ label, description, children }: { label: string; description: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-4">
      <div className="flex flex-col gap-0.5">
        <span className="text-xs font-medium">{label}</span>
        <span className="text-2xs text-muted-foreground">{description}</span>
      </div>
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
        <AppearanceRow label="Color mode" description="Overall color scheme of the interface">
          <SegmentedControl options={COLOR_MODES} value={colorMode} onChange={setColorMode} />
        </AppearanceRow>

        <AppearanceRow label="Accent color" description="Primary color used for buttons, links, and highlights">
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
        </AppearanceRow>

        <AppearanceRow label="Corner style" description="Shape of toggle switches and segmented controls">
          <SegmentedControl options={CORNER_STYLES} value={cornerStyle} onChange={setCornerStyle} />
        </AppearanceRow>

        <AppearanceRow label="Border radius" description="Roundness of cards, inputs, and panels">
          <div className="flex items-center gap-2">
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={borderRadius}
              onChange={(e) => setBorderRadius(parseFloat(e.target.value))}
              className="w-24 accent-primary"
            />
            <span className="text-xs text-muted-foreground w-12 text-right">{borderRadius.toFixed(2)}rem</span>
          </div>
        </AppearanceRow>

        <AppearanceRow label="UI scale" description="Base font size — all spacing and controls scale proportionally">
          <div className="flex items-center gap-2">
            <input
              type="range"
              min={12}
              max={20}
              step={1}
              value={uiScale}
              onChange={(e) => setUiScale(parseInt(e.target.value, 10))}
              className="w-24 accent-primary"
            />
            <span className="text-xs text-muted-foreground w-8 text-right">{uiScale}px</span>
          </div>
        </AppearanceRow>

        <AppearanceRow label="Canvas label scale" description="Size of frame labels and floating control panels on the canvas">
          <div className="flex items-center gap-2">
            <input
              type="range"
              min={0.5}
              max={2}
              step={0.1}
              value={canvasLabelScale}
              onChange={(e) => setCanvasLabelScale(parseFloat(e.target.value))}
              className="w-24 accent-primary"
            />
            <span className="text-xs text-muted-foreground w-8 text-right">{canvasLabelScale.toFixed(1)}x</span>
          </div>
        </AppearanceRow>
      </div>
    </div>
  );
}

function ConnectionPanel() {
  const backendUrl = useConnectionStore((s) => s.backendUrl);
  const username = useConnectionStore((s) => s.username);
  const password = useConnectionStore((s) => s.password);
  const storeSetUrl = useConnectionStore((s) => s.setBackendUrl);
  const storeSetAuth = useConnectionStore((s) => s.setAuth);
  const storeReset = useConnectionStore((s) => s.reset);

  const [urlInput, setUrlInput] = useState(backendUrl);
  const [userInput, setUserInput] = useState(username);
  const [passInput, setPassInput] = useState(password);
  const [status, setStatus] = useState<"idle" | "checking" | "connected" | "unreachable">("idle");

  // Sync inputs when store changes externally (e.g. reset)
  useEffect(() => { setUrlInput(backendUrl); }, [backendUrl]);
  useEffect(() => { setUserInput(username); }, [username]);
  useEffect(() => { setPassInput(password); }, [password]);

  const checkConnection = useCallback(async (baseUrl: string) => {
    setStatus("checking");
    try {
      await fetch(`${baseUrl}/sdapi/v2/server-info`, { signal: AbortSignal.timeout(5000) });
      setStatus("connected");
    } catch {
      setStatus("unreachable");
    }
  }, []);

  // Check connection on mount if a custom URL is stored
  useEffect(() => {
    if (backendUrl) {
      checkConnection(backendUrl);
    } else {
      setStatus("connected"); // default origin is always "connected"
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const handleConnect = useCallback(async () => {
    const effectiveUrl = urlInput.replace(/\/$/, "") || window.location.origin;
    api.setBaseUrl(effectiveUrl);
    if (userInput && passInput) {
      api.setAuth(userInput, passInput);
    } else {
      api.clearAuth();
    }
    storeSetUrl(urlInput.replace(/\/$/, ""));
    storeSetAuth(userInput, passInput);
    ws.updateUrl(api.getWebSocketUrl("/sdapi/v2/ws"));
    queryClient.invalidateQueries();
    await checkConnection(effectiveUrl);
    toast.success("Connection updated");
  }, [urlInput, userInput, passInput, storeSetUrl, storeSetAuth, checkConnection]);

  const handleReset = useCallback(() => {
    storeReset();
    api.setBaseUrl(window.location.origin);
    api.clearAuth();
    ws.updateUrl(api.getWebSocketUrl("/sdapi/v2/ws"));
    queryClient.invalidateQueries();
    setStatus("connected");
    toast.success("Connection reset to default");
  }, [storeReset]);

  return (
    <div>
      <h3 className="text-sm font-medium mb-4">Connection</h3>
      <div className="space-y-4">
        <AppearanceRow label="Backend URL" description="Leave empty to use the current origin">
          <div className="flex items-center gap-2">
            <Input
              value={urlInput}
              onChange={(e) => setUrlInput(e.target.value)}
              placeholder={window.location.origin}
              className="h-7 w-56 text-xs font-mono"
            />
            {status === "checking" && <span className="text-xs text-muted-foreground">Checking...</span>}
            {status === "connected" && <span className="text-xs text-emerald-500">Connected</span>}
            {status === "unreachable" && <span className="text-xs text-destructive">Unreachable</span>}
          </div>
        </AppearanceRow>

        <AppearanceRow label="Username" description="For basic auth (optional)">
          <Input
            value={userInput}
            onChange={(e) => setUserInput(e.target.value)}
            placeholder="username"
            className="h-7 w-40 text-xs"
            autoComplete="username"
          />
        </AppearanceRow>

        <AppearanceRow label="Password" description="For basic auth (optional)">
          <Input
            type="password"
            value={passInput}
            onChange={(e) => setPassInput(e.target.value)}
            placeholder="password"
            className="h-7 w-40 text-xs"
            autoComplete="current-password"
          />
        </AppearanceRow>

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
      for (const [key, info] of Object.entries(meta)) {
        if (info.section_id !== section.id) continue;
        if (!info.visible || info.hidden || info.is_legacy) continue;
        if (GRADIO_ONLY_KEYS.has(key)) continue;
        if (info.component === "separator") {
          if (info.label) settings.push({ key, label: info.label, component: "separator" });
          continue;
        }
        if (!(key in options)) continue;

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

  const backendReady = !isLoading && !isInfoLoading && !!options && !!optionsInfo;

  return (
    <div className="flex h-full">
      {/* Section navigation */}
      <div className="w-48 border-r border-border flex-shrink-0 flex flex-col">
        <div className="p-2">
          <div className="relative">
            <Search size={14} className="absolute left-2 top-1/2 -translate-y-1/2 text-muted-foreground" />
            <Input
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
