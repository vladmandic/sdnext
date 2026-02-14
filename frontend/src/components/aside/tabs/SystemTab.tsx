import { useGpuStatus, useMemory, useLoadedModels, useServerInfo } from "@/api/hooks/useServer";
import { GroupedModels } from "@/components/layout/LoadedModelsPanel";
import { formatBytes } from "@/lib/utils";

export function SystemTab() {
  const { data: gpus } = useGpuStatus();
  const { data: memory } = useMemory();
  const { data: models } = useLoadedModels();
  const { data: serverInfo } = useServerInfo();

  const gpu = gpus?.[0];
  const gpuData = gpu?.data as Record<string, number> | undefined;
  const vramAllocated = memory?.cuda?.allocated?.current;
  const vramTotal = memory?.cuda?.system?.total;
  const ramUsed = memory?.ram?.used;
  const ramTotal = memory?.ram?.total;

  return (
    <div className="p-3 space-y-4">
      {/* GPU */}
      {gpu && (
        <Section title="GPU">
          <Row label="Name" value={gpu.name} />
          {gpuData?.temperature != null && <BarRow label="Temp" value={gpuData.temperature} max={100} unit="C" />}
          {gpuData?.utilization != null && <BarRow label="Load" value={gpuData.utilization} max={100} unit="%" />}
          {gpuData?.power != null && (
            <Row label="Power" value={`${Math.round(gpuData.power)}W${gpuData.power_limit ? ` / ${Math.round(gpuData.power_limit)}W` : ""}`} />
          )}
        </Section>
      )}

      {/* Memory */}
      <Section title="Memory">
        {vramAllocated != null && vramTotal != null && vramTotal > 0 && (
          <BarRow label="VRAM" value={vramAllocated} max={vramTotal} formatter={formatBytes} />
        )}
        {ramUsed != null && ramTotal != null && ramTotal > 0 && (
          <BarRow label="RAM" value={ramUsed} max={ramTotal} formatter={formatBytes} />
        )}
        {!vramTotal && !ramTotal && <p className="text-xs text-muted-foreground">No memory data</p>}
      </Section>

      {/* Loaded Models */}
      <Section title={`Loaded Models (${models?.length ?? 0})`}>
        {models && models.length > 0 ? (
          <GroupedModels models={models} />
        ) : (
          <p className="text-xs text-muted-foreground">No models loaded</p>
        )}
      </Section>

      {/* Server */}
      {serverInfo && (
        <Section title="Server">
          <Row label="Version" value={serverInfo.version?.app} />
          <Row label="Backend" value={serverInfo.backend} />
          <Row label="Platform" value={serverInfo.platform} />
        </Section>
      )}
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <h3 className="text-xs font-medium text-muted-foreground mb-2">{title}</h3>
      <div className="space-y-1.5">{children}</div>
    </div>
  );
}

function Row({ label, value }: { label: string; value?: string | number | null }) {
  if (value == null) return null;
  return (
    <div className="flex justify-between gap-2 text-xs">
      <span className="text-muted-foreground">{label}</span>
      <span className="text-right truncate">{value}</span>
    </div>
  );
}

interface BarRowProps {
  label: string;
  value: number;
  max: number;
  unit?: string;
  formatter?: (n: number) => string;
}

function BarRow({ label, value, max, unit, formatter }: BarRowProps) {
  const pct = max > 0 ? Math.min(100, (value / max) * 100) : 0;
  const fmt = formatter ?? ((n: number) => `${Math.round(n)}${unit ?? ""}`);
  return (
    <div className="space-y-0.5">
      <div className="flex justify-between text-xs">
        <span className="text-muted-foreground">{label}</span>
        <span className="tabular-nums">{fmt(value)} / {fmt(max)}</span>
      </div>
      <div className="h-1.5 rounded-full bg-muted overflow-hidden">
        <div
          className="h-full rounded-full bg-accent transition-all duration-300"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
