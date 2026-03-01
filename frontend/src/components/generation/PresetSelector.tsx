import { useState, useCallback, useMemo, useRef } from "react";
import { Save, Download, Upload, Trash2 } from "lucide-react";
import { toast } from "sonner";
import { Combobox } from "@/components/ui/combobox";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { useGenerationStore } from "@/stores/generationStore";
import { usePresetStore, snapshotParams } from "@/stores/presetStore";
import { useCurrentCheckpoint } from "@/api/hooks/useModels";

export function PresetSelector() {
  const [saveOpen, setSaveOpen] = useState(false);
  const [saveName, setSaveName] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const { data: checkpoint } = useCurrentCheckpoint();
  const modelType = checkpoint?.type ?? null;

  const getPresetsForModelType = usePresetStore((s) => s.getPresetsForModelType);
  const addPreset = usePresetStore((s) => s.addPreset);
  const removePreset = usePresetStore((s) => s.removePreset);
  const importPresets = usePresetStore((s) => s.importPresets);
  const exportPresets = usePresetStore((s) => s.exportPresets);
  const setParams = useGenerationStore((s) => s.setParams);

  const presets = getPresetsForModelType(modelType);
  const presetIds = useMemo(() => presets.map((p) => p.id), [presets]);
  const { presetNameMap, presetBuiltInMap } = useMemo(() => {
    const nameMap: Record<string, string> = {};
    const builtInMap: Record<string, boolean> = {};
    for (const p of presets) {
      nameMap[p.id] = p.name;
      builtInMap[p.id] = !!p.builtIn;
    }
    return { presetNameMap: nameMap, presetBuiltInMap: builtInMap };
  }, [presets]);

  const handleLoad = useCallback(
    (id: string) => {
      const preset = presets.find((p) => p.id === id);
      if (!preset) return;
      setParams(preset.params);
      toast.success(`Loaded preset: ${preset.name}`);
    },
    [presets, setParams],
  );

  const handleSave = useCallback(() => {
    const name = saveName.trim();
    if (!name) return;
    const params = snapshotParams();
    addPreset({ name, compatibleTypes: [], params });
    setSaveName("");
    setSaveOpen(false);
    toast.success(`Saved preset: ${name}`);
  }, [saveName, addPreset]);

  const handleDelete = useCallback(
    (id: string) => {
      if (presetBuiltInMap[id]) return;
      const preset = presets.find((p) => p.id === id);
      removePreset(id);
      toast.success(`Deleted preset: ${preset?.name ?? "preset"}`);
    },
    [presets, presetBuiltInMap, removePreset],
  );

  const handleExport = useCallback(() => {
    const json = exportPresets();
    navigator.clipboard.writeText(json);
    toast.success("Presets copied to clipboard");
  }, [exportPresets]);

  const handleImportClick = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  const handleImportFile = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = () => {
        const count = importPresets(reader.result as string);
        if (count > 0) toast.success(`Imported ${count} preset${count > 1 ? "s" : ""}`);
        else toast.error("No valid presets found in file");
      };
      reader.readAsText(file);
      e.target.value = "";
    },
    [importPresets],
  );

  return (
    <div className="flex items-center gap-1">
      <Combobox
        value=""
        onValueChange={handleLoad}
        options={presetIds}
        placeholder="Presets..."
        renderLabel={(_v, label) => {
          const name = presetNameMap[_v] ?? label;
          const isBuiltIn = presetBuiltInMap[_v];
          return (
            <span className="flex items-center gap-1.5">
              <span className="truncate">{name}</span>
              {isBuiltIn && <span className="text-4xs font-medium bg-muted px-1 rounded shrink-0">built-in</span>}
              {!isBuiltIn && (
                <button
                  type="button"
                  className="ml-auto opacity-0 group-hover:opacity-100 hover:text-destructive shrink-0"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleDelete(_v);
                  }}
                >
                  <Trash2 size={12} />
                </button>
              )}
            </span>
          );
        }}
        className="h-6 text-2xs w-28"
      />

      <Popover open={saveOpen} onOpenChange={setSaveOpen}>
        <PopoverTrigger asChild>
          <Button variant="ghost" size="icon-sm" title="Save current settings as preset">
            <Save size={14} />
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-56 p-2">
          <div className="flex gap-1">
            <Input
              value={saveName}
              onChange={(e) => setSaveName(e.target.value)}
              placeholder="Preset name..."
              className="h-6 text-2xs flex-1"
              onKeyDown={(e) => {
                if (e.key === "Enter") handleSave();
              }}
            />
            <Button size="sm" onClick={handleSave} disabled={!saveName.trim()} className="h-6 text-2xs px-2">
              Save
            </Button>
          </div>
        </PopoverContent>
      </Popover>

      <Button variant="ghost" size="icon-sm" onClick={handleImportClick} title="Import presets from file">
        <Download size={14} />
      </Button>
      <Button variant="ghost" size="icon-sm" onClick={handleExport} title="Export presets to clipboard">
        <Upload size={14} />
      </Button>

      <input ref={fileInputRef} type="file" accept=".json" className="hidden" onChange={handleImportFile} />
    </div>
  );
}
