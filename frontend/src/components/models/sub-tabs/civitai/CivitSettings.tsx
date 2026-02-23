import { useState } from "react";
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
          <SubfolderSection value={settings.save_subfolder} onSave={(v) => save.mutate({ save_subfolder: v })} isPending={save.isPending} />
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

function SubfolderSection({ value, onSave, isPending }: { value: string; onSave: (v: string) => void; isPending: boolean }) {
  const [draft, setDraft] = useState(value);
  const changed = draft !== value;

  return (
    <div className="space-y-1">
      <Label htmlFor="subfolder-template" className="text-xs">Save subfolder template</Label>
      <div className="flex gap-2">
        <Input id="subfolder-template" placeholder="{{BASEMODEL}}/{{MODELNAME}}" value={draft} onChange={(e) => setDraft(e.target.value)} className="h-6 text-2xs" />
        {changed && <Button size="sm" className="h-6 text-2xs" disabled={isPending} onClick={() => onSave(draft)}>Save</Button>}
      </div>
      <p className="text-3xs text-muted-foreground">
        {"Variables: {{BASEMODEL}}, {{MODELNAME}}, {{CREATOR}}, {{TYPE}}, {{NSFW}}, {{MODELID}}, {{VERSIONID}}, {{VERSIONNAME}}"}
      </p>
    </div>
  );
}
