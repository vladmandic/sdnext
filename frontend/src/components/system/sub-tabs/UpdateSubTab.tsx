import { useState } from "react";
import { Loader2, ExternalLink, Check, AlertCircle } from "lucide-react";
import { useUpdateCheck, useApplyUpdate } from "@/api/hooks/useSystem";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Section, Row } from "./OverviewSubTab";

export function UpdateSubTab() {
  const { data: info, isFetching, refetch } = useUpdateCheck();
  const applyUpdate = useApplyUpdate();
  const [rebase, setRebase] = useState(true);
  const [submodules, setSubmodules] = useState(true);
  const [extensions, setExtensions] = useState(true);

  return (
    <div className="space-y-4">
      <Button size="sm" onClick={() => refetch()} disabled={isFetching} className="w-full">
        {isFetching ? <Loader2 className="h-3.5 w-3.5 animate-spin mr-1.5" /> : null}
        Check for updates
      </Button>

      {info && !info.error && (
        <Section title="Version Info">
          <Row label="Branch" value={info.branch} />
          <Row label="Current" value={`${info.current_date} (${info.current_hash})`} />
          <Row label="Latest" value={`${info.latest_date} (${info.latest_hash})`} />
          <div className="flex items-center gap-1.5 text-xs mt-1">
            {info.up_to_date ? (
              <>
                <Check className="h-3.5 w-3.5 text-green-500" />
                <span className="text-green-500">Up to date</span>
              </>
            ) : (
              <>
                <AlertCircle className="h-3.5 w-3.5 text-yellow-500" />
                <span className="text-yellow-500">Update available</span>
              </>
            )}
          </div>
          {info.url && (
            <a href={info.url} target="_blank" rel="noopener noreferrer" className="flex items-center gap-1 text-xs text-accent hover:underline mt-1">
              <ExternalLink className="h-3 w-3" />
              View on GitHub
            </a>
          )}
        </Section>
      )}

      {info?.error && (
        <p className="text-xs text-destructive">{info.error}</p>
      )}

      <Section title="Download Options">
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Checkbox id="rebase" checked={rebase} onCheckedChange={(c) => setRebase(c === true)} />
            <Label htmlFor="rebase" className="text-xs">Rebase</Label>
          </div>
          <div className="flex items-center gap-2">
            <Checkbox id="submodules" checked={submodules} onCheckedChange={(c) => setSubmodules(c === true)} />
            <Label htmlFor="submodules" className="text-xs">Submodules</Label>
          </div>
          <div className="flex items-center gap-2">
            <Checkbox id="extensions" checked={extensions} onCheckedChange={(c) => setExtensions(c === true)} />
            <Label htmlFor="extensions" className="text-xs">Extensions</Label>
          </div>
        </div>
      </Section>

      <Button
        size="sm"
        onClick={() => applyUpdate.mutate({ rebase, submodules, extensions })}
        disabled={applyUpdate.isPending}
        className="w-full"
      >
        {applyUpdate.isPending ? <Loader2 className="h-3.5 w-3.5 animate-spin mr-1.5" /> : null}
        Download updates
      </Button>

      {applyUpdate.data && (
        <div className="text-xs space-y-1 p-2 rounded bg-muted">
          <p className="font-medium">{applyUpdate.data.changed ? "Update applied" : "No changes"}</p>
          <pre className="whitespace-pre-wrap text-muted-foreground text-[10px]">{applyUpdate.data.status}</pre>
        </div>
      )}
    </div>
  );
}
