import { useState } from "react";
import { Settings, Check, ChevronDown, ChevronRight, X } from "lucide-react";
import { useHfSettings, useHfSaveSettings, useHfMe } from "@/api/hooks/useHuggingface";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

function TokenSection({ configured }: { configured: boolean }) {
  const [editing, setEditing] = useState(false);
  const [saved, setSaved] = useState(false);
  const [value, setValue] = useState("");
  const [error, setError] = useState("");
  const save = useHfSaveSettings();
  const { data: profile } = useHfMe(configured || saved);

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
        {profile?.avatar ? (
          <img src={profile.avatar} alt="" className="h-6 w-6 rounded-full shrink-0 object-cover" />
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
      <Label htmlFor="hf-token" className="text-xs">HuggingFace API token</Label>
      <div className="flex gap-2">
        <Input id="hf-token" placeholder="Paste token..." value={value} onChange={(e) => { setValue(e.target.value); setError(""); }} autoComplete="off" className="h-6 text-2xs" />
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

export function HfSettings() {
  const [open, setOpen] = useState(false);
  const { data: settings } = useHfSettings();

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
        </div>
      )}
    </div>
  );
}
