import { useState, useRef, useMemo } from "react";
import { Settings, Check, ChevronDown, ChevronRight, X } from "lucide-react";
import { useCivitSettings, useCivitSaveSettings, useCivitMe } from "@/api/hooks/useCivitai";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";

export function CivitSettings() {
  const [open, setOpen] = useState(false);
  const { data: settings } = useCivitSettings();
  const save = useCivitSaveSettings();

  if (!settings) return null;

  return (
    <div className="border border-border/50 rounded-md">
      <button type="button" onClick={() => setOpen(!open)} className="flex items-center gap-2 w-full px-3 py-2 text-left hover:bg-muted/30">
        <Settings className="h-3 w-3 shrink-0 text-muted-foreground" />
        <span className="text-xs font-medium">Settings</span>
        {open ? <ChevronDown className="h-3 w-3 ml-auto shrink-0 text-muted-foreground" /> : <ChevronRight className="h-3 w-3 ml-auto shrink-0 text-muted-foreground" />}
      </button>
      {open && (
        <div className="px-3 pb-3 space-y-4">
          <TokenSection configured={settings.token_configured} />
          <div className="flex items-center justify-between gap-2">
            <Label htmlFor="subfolder-enabled" className="text-xs">Save downloads into subfolders</Label>
            <Switch id="subfolder-enabled" size="sm" checked={settings.save_subfolder_enabled} onCheckedChange={(v) => save.mutate({ save_subfolder_enabled: v })} />
          </div>
          {settings.save_subfolder_enabled && (
            <SubfolderSection value={settings.save_subfolder} onSave={(v) => save.mutate({ save_subfolder: v })} isPending={save.isPending} />
          )}
          <div className="flex items-center justify-between gap-2">
            <Label htmlFor="hash-mismatch" className="text-xs">Discard downloads with hash mismatch</Label>
            <Switch id="hash-mismatch" size="sm" checked={settings.discard_hash_mismatch} onCheckedChange={(v) => save.mutate({ discard_hash_mismatch: v })} />
          </div>
        </div>
      )}
    </div>
  );
}

function TokenSection({ configured }: { configured: boolean }) {
  const [editing, setEditing] = useState(false);
  const [saved, setSaved] = useState(false);
  const [value, setValue] = useState("");
  const [error, setError] = useState("");
  const save = useCivitSaveSettings();
  const { data: profile } = useCivitMe(configured || saved);

  function handleSave() {
    setError("");
    save.mutate({ token: value.trim() }, {
      onSuccess: () => { setValue(""); setSaved(true); setEditing(false); },
      onError: (err) => { setError(err instanceof Error ? err.message : "Invalid token"); },
    });
  }

  if ((configured || saved) && !editing) {
    return (
      <div className="flex items-center gap-2">
        <Check className="h-3 w-3 text-green-500 shrink-0" />
        {profile?.profilePicture || profile?.image ? (
          <img src={profile.profilePicture ?? profile.image!} alt="" className="h-6 w-6 rounded-full shrink-0 object-cover" />
        ) : null}
        <span className="text-xs text-muted-foreground">
          {profile?.username ? `Signed in as ${profile.username}` : "API token configured"}
        </span>
        <Button size="sm" variant="ghost" className="h-6 text-3xs ml-auto" onClick={() => { setEditing(true); setSaved(false); }}>Change</Button>
      </div>
    );
  }

  return (
    <div className="space-y-1">
      <Label htmlFor="civit-token" className="text-xs">CivitAI API token</Label>
      <div className="flex gap-2">
        <Input id="civit-token" placeholder="Paste token..." value={value} onChange={(e) => { setValue(e.target.value); setError(""); }} autoComplete="off" className="h-6 text-2xs" />
        <Button size="sm" className="h-6 text-2xs" disabled={!value.trim() || save.isPending} onClick={handleSave}>{save.isPending ? "Validating..." : "Save"}</Button>
      </div>
      {error && (
        <div className="flex items-center gap-1 text-3xs text-destructive">
          <X className="h-3 w-3 shrink-0" />
          <span>{error}</span>
        </div>
      )}
    </div>
  );
}

const TEMPLATE_VARS = [
  { key: "BASEMODEL", example: "SDXL" },
  { key: "MODELNAME", example: "My_Model" },
  { key: "CREATOR", example: "artist42" },
  { key: "TYPE", example: "LORA" },
  { key: "VERSIONNAME", example: "v2.0" },
  { key: "NSFW", example: "sfw" },
  { key: "MODELID", example: "12345" },
  { key: "VERSIONID", example: "67890" },
] as const;

function resolvePreview(template: string): string {
  if (!template) return "";
  let result = template;
  for (const v of TEMPLATE_VARS) {
    result = result.replaceAll(`{{${v.key}}}`, v.example);
  }
  return result.replace(/[/\\]+/g, "/").replace(/^\/|\/$/g, "");
}

function SubfolderSection({ value, onSave, isPending }: { value: string; onSave: (v: string) => void; isPending: boolean }) {
  const [draft, setDraft] = useState(value);
  const inputRef = useRef<HTMLInputElement>(null);
  const changed = draft !== value;
  const preview = useMemo(() => resolvePreview(draft), [draft]);

  function insertVar(key: string) {
    const el = inputRef.current;
    if (!el) {
      setDraft((d) => d + `{{${key}}}`);
      return;
    }
    const tag = `{{${key}}}`;
    const start = el.selectionStart ?? draft.length;
    const end = el.selectionEnd ?? start;
    const next = draft.slice(0, start) + tag + draft.slice(end);
    setDraft(next);
    requestAnimationFrame(() => {
      const pos = start + tag.length;
      el.setSelectionRange(pos, pos);
      el.focus();
    });
  }

  return (
    <div className="space-y-1.5">
      <Label htmlFor="subfolder-template" className="text-xs">Save subfolder template</Label>
      <div className="flex gap-2">
        <Input ref={inputRef} id="subfolder-template" placeholder="e.g. {{BASEMODEL}}/{{CREATOR}}" value={draft} onChange={(e) => setDraft(e.target.value)} className="h-6 text-2xs font-mono" />
        {changed && <Button size="sm" className="h-6 text-2xs" disabled={isPending} onClick={() => onSave(draft)}>Save</Button>}
      </div>
      <div className="flex flex-wrap gap-1">
        {TEMPLATE_VARS.map((v) => (
          <button
            key={v.key}
            type="button"
            onClick={() => insertVar(v.key)}
            className="px-1.5 py-0.5 rounded text-3xs font-mono bg-muted text-muted-foreground hover:bg-accent hover:text-accent-foreground transition-colors"
          >
            {v.key}
          </button>
        ))}
      </div>
      {preview && (
        <p className="text-3xs text-muted-foreground">
          <span className="font-medium text-foreground/70">Preview: </span>
          <span className="font-mono">.../Lora/{preview}/</span>
        </p>
      )}
    </div>
  );
}
