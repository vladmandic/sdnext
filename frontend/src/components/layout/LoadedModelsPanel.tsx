import { useLoadedModels, useMemory } from "@/api/hooks/useServer";
import type { LoadedModel } from "@/api/types/server";
import { formatBytes } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";

const CATEGORY_ORDER = [
  "pipeline",
  "component",
  "controlnet",
  "t2iadapter",
  "lora",
  "lora_cached",
  "ipadapter",
  "upscaler",
  "caption",
  "enhance",
  "detailer",
] as const;

const CATEGORY_LABELS: Record<string, string> = {
  pipeline: "Pipeline",
  component: "Components",
  controlnet: "ControlNet",
  t2iadapter: "T2I Adapter",
  lora: "LoRA (Active)",
  lora_cached: "LoRA (Cached)",
  ipadapter: "IP-Adapter",
  upscaler: "Upscaler",
  caption: "Caption",
  enhance: "Prompt Enhance",
  detailer: "Detailer",
};

export function DeviceBadge({ device }: { device?: string | null }) {
  if (!device) return null;
  const variant = device.startsWith("cuda")
    ? "default"
    : device === "cpu"
      ? "secondary"
      : "outline";
  return (
    <Badge variant={variant} className="text-3xs px-1.5 py-0">
      {device}
    </Badge>
  );
}

export function DtypeLabel({ model }: { model: LoadedModel }) {
  const quant = model.extra?.quant as string | undefined;
  const dtype = model.dtype;
  if (!quant && !dtype) return null;
  const label = quant ? `${quant} ${dtype ?? ""}`.trim() : dtype!;
  return <span className="text-3xs text-muted-foreground">{label}</span>;
}

export function ModelRow({ model }: { model: LoadedModel }) {
  return (
    <div className="flex items-center justify-between gap-2 py-1 px-2 rounded hover:bg-muted/50">
      <span className="font-mono text-xs truncate min-w-0">{model.name}</span>
      <div className="flex items-center gap-1.5 shrink-0">
        <DtypeLabel model={model} />
        {model.size_bytes != null && model.size_bytes > 0 && (
          <span className="text-3xs text-muted-foreground tabular-nums">
            {formatBytes(model.size_bytes)}
          </span>
        )}
        <DeviceBadge device={model.device} />
      </div>
    </div>
  );
}

export function GroupedModels({ models }: { models: LoadedModel[] }) {
  const groups = new Map<string, LoadedModel[]>();
  for (const m of models) {
    const list = groups.get(m.category) ?? [];
    list.push(m);
    groups.set(m.category, list);
  }

  return (
    <>
      {CATEGORY_ORDER.filter((c) => groups.has(c)).map((category) => (
        <div key={category} className="mb-3 last:mb-0">
          <h4 className="text-xs font-medium text-muted-foreground mb-1 px-2">
            {CATEGORY_LABELS[category] ?? category}
          </h4>
          {groups.get(category)!.map((model, i) => (
            <ModelRow key={`${model.category}-${model.name}-${i}`} model={model} />
          ))}
        </div>
      ))}
    </>
  );
}

export function LoadedModelsPanel({ children }: { children: React.ReactNode }) {
  const { data: models } = useLoadedModels();
  const { data: memory } = useMemory();

  const modelCount = models?.length ?? 0;
  const totalSize = models?.reduce((acc, m) => acc + (m.size_bytes ?? 0), 0) ?? 0;
  const vramAllocated = memory?.cuda?.allocated?.current;
  const vramTotal = memory?.cuda?.system?.total;

  return (
    <Dialog>
      <DialogTrigger asChild>{children}</DialogTrigger>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Loaded Models</DialogTitle>
          <DialogDescription>
            {modelCount} model{modelCount !== 1 ? "s" : ""} loaded
            {totalSize > 0 && <> &middot; {formatBytes(totalSize)} params</>}
            {vramAllocated != null && vramTotal != null && (
              <> &middot; VRAM {formatBytes(vramAllocated)} / {formatBytes(vramTotal)}</>
            )}
          </DialogDescription>
        </DialogHeader>
        <ScrollArea className="max-h-[60vh]">
          {modelCount > 0 ? (
            <GroupedModels models={models!} />
          ) : (
            <p className="text-sm text-muted-foreground text-center py-8">
              No models currently loaded
            </p>
          )}
        </ScrollArea>
      </DialogContent>
    </Dialog>
  );
}
