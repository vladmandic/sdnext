import { Loader2, RefreshCw } from "lucide-react";
import { useSystemInfoFull } from "@/api/hooks/useSystem";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Section, Row } from "./OverviewSubTab";

export function SystemInfoSubTab() {
  const { data, isFetching, refetch } = useSystemInfoFull();

  return (
    <div className="space-y-4">
      <Button size="sm" onClick={() => refetch()} disabled={isFetching} className="w-full">
        {isFetching ? <Loader2 className="h-3.5 w-3.5 animate-spin mr-1.5" /> : <RefreshCw className="h-3.5 w-3.5 mr-1.5" />}
        {data ? "Refresh" : "Load System Info"}
      </Button>

      {!data && !isFetching && (
        <p className="text-xs text-muted-foreground text-center py-4">Click button to load full system info</p>
      )}

      {data && (
        <>
          {data.version && (
            <Section title="Version">
              {Object.entries(data.version).map(([k, v]) => (
                <Row key={k} label={k} value={v} />
              ))}
            </Section>
          )}

          {data.uptime && <Row label="Uptime" value={data.uptime} />}
          {data.timestamp && <Row label="Timestamp" value={data.timestamp} />}

          {data.platform && (
            <Section title="Platform">
              {Object.entries(data.platform).map(([k, v]) => (
                <Row key={k} label={k} value={v} />
              ))}
            </Section>
          )}

          {data.torch && <Row label="Torch" value={data.torch} />}

          {data.gpu && (
            <Section title="GPU">
              {Object.entries(data.gpu).map(([k, v]) => (
                <Row key={k} label={k} value={v} />
              ))}
            </Section>
          )}

          {data.device && (
            <Section title="Device">
              {Object.entries(data.device).map(([k, v]) => (
                <Row key={k} label={k} value={v} />
              ))}
            </Section>
          )}

          {data.libs && (
            <Section title="Libraries">
              {Object.entries(data.libs).map(([k, v]) => (
                <Row key={k} label={k} value={v} />
              ))}
            </Section>
          )}

          <Row label="Backend" value={data.backend} />
          <Row label="Pipeline" value={data.pipeline} />
          <Row label="Cross Attention" value={data.cross_attention} />

          {data.flags && data.flags.length > 0 && (
            <Section title="Flags">
              <div className="flex flex-wrap gap-1">
                {data.flags.map((flag) => (
                  <Badge key={flag} variant="secondary" className="text-3xs">{flag}</Badge>
                ))}
              </div>
            </Section>
          )}
        </>
      )}
    </div>
  );
}
