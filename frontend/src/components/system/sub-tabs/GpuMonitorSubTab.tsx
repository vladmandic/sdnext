import { useCallback, useEffect, useSyncExternalStore } from "react";
import { useGpuStatus } from "@/api/hooks/useServer";
import { Section, Row } from "./OverviewSubTab";

const MAX_POINTS = 60;

const chartStore = {
  gpu: [] as number[],
  vram: [] as number[],
  _lastId: 0,
  _listeners: new Set<() => void>(),
  push(chart: [number, number], id: number) {
    if (id === this._lastId) return;
    this._lastId = id;
    this.gpu = [...this.gpu, chart[1]].slice(-MAX_POINTS);
    this.vram = [...this.vram, chart[0]].slice(-MAX_POINTS);
    for (const l of this._listeners) l();
  },
  subscribe(listener: () => void) {
    chartStore._listeners.add(listener);
    return () => { chartStore._listeners.delete(listener); };
  },
};

function Sparkline({ data, color, height = 40 }: { data: number[]; color: string; height?: number }) {
  if (data.length < 2) return null;
  const width = 200;
  const padY = 2;
  const max = 100;
  const points = data
    .map((v, i) => {
      const x = (i / (MAX_POINTS - 1)) * width;
      const y = height - padY - ((v / max) * (height - padY * 2));
      return `${x},${y}`;
    })
    .join(" ");

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full" preserveAspectRatio="none">
      <polyline
        points={points}
        fill="none"
        stroke={color}
        strokeWidth="1.5"
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </svg>
  );
}

export function GpuMonitorSubTab() {
  const { data: gpus, dataUpdatedAt } = useGpuStatus();
  const gpu = gpus?.[0];
  const chart = gpu?.chart as [number, number] | undefined;

  useEffect(() => {
    if (chart) chartStore.push(chart, dataUpdatedAt);
  }, [chart, dataUpdatedAt]);

  const gpuHistory = useSyncExternalStore(chartStore.subscribe, useCallback(() => chartStore.gpu, []));
  const vramHistory = useSyncExternalStore(chartStore.subscribe, useCallback(() => chartStore.vram, []));
  const gpuData = gpu?.data as Record<string, string> | undefined;

  return (
    <div className="space-y-4">
      {gpu && (
        <>
          <Section title={gpu.name}>
            <div className="space-y-2">
              <div>
                <div className="flex justify-between text-3xs text-muted-foreground mb-0.5">
                  <span>GPU Load</span>
                  <span className="tabular-nums">{chart?.[1] ?? 0}%</span>
                </div>
                <div className="border border-border rounded overflow-hidden">
                  <Sparkline data={gpuHistory} color="hsl(var(--accent))" />
                </div>
              </div>
              <div>
                <div className="flex justify-between text-3xs text-muted-foreground mb-0.5">
                  <span>VRAM Load</span>
                  <span className="tabular-nums">{chart?.[0] ?? 0}%</span>
                </div>
                <div className="border border-border rounded overflow-hidden">
                  <Sparkline data={vramHistory} color="hsl(var(--chart-2, var(--accent)))" />
                </div>
              </div>
            </div>
          </Section>

          {gpuData && (
            <Section title="Details">
              {Object.entries(gpuData).map(([key, val]) => (
                <Row key={key} label={key} value={String(val)} />
              ))}
            </Section>
          )}
        </>
      )}

      {!gpu && (
        <p className="text-xs text-muted-foreground text-center py-4">No GPU data available</p>
      )}
    </div>
  );
}
