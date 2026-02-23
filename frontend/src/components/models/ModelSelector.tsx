import { useModelList, useLoadModel, useRefreshModels, useReloadModel, useUnloadModel, useCurrentCheckpoint, useIsModelLoading } from "@/api/hooks/useModels";
import { useOptions } from "@/api/hooks/useSettings";
import { RefreshCw, Check, ChevronsUpDown, ArrowBigDownDash, FolderSync } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from "@/components/ui/command";

function formatPipelineClass(cls: string | null | undefined): string | null {
  if (!cls) return null;
  return cls.replace(/Pipeline$/, "").replace(/Img2Img$|Inpaint$/, "");
}

export function ModelSelector() {
  const { data: models } = useModelList();
  const { data: options } = useOptions();
  const { data: checkpoint } = useCurrentCheckpoint();
  const loadModel = useLoadModel();
  const reloadModel = useReloadModel();
  const unloadModel = useUnloadModel();
  const refreshModels = useRefreshModels();
  const isModelLoading = useIsModelLoading();

  const [open, setOpen] = useState(false);

  const currentModel = (options?.sd_model_checkpoint as string) ?? "No model loaded";
  const pipelineClass = formatPipelineClass(checkpoint?.class);

  async function handleSelect(title: string) {
    setOpen(false);
    try {
      await loadModel.mutateAsync(title);
      toast.success("Model loaded", { description: title });
    } catch (err) {
      toast.error("Failed to load model", { description: err instanceof Error ? err.message : String(err) });
    }
  }

  return (
    <div className="flex items-center gap-1">
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <Button
            variant="secondary"
            size="sm"
            disabled={isModelLoading}
            className={cn("w-full max-w-md justify-between text-xs h-7 px-2", isModelLoading && "opacity-60")}
          >
            <span className="flex items-center gap-2 truncate">
              {isModelLoading && <RefreshCw size={12} className="animate-spin flex-shrink-0" />}
              <span className="truncate">{currentModel}</span>
            </span>
            <ChevronsUpDown size={12} className="flex-shrink-0 opacity-50" />
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-[25rem] p-0" align="start">
          <Command>
            <CommandInput placeholder="Search models..." />
            <CommandList>
              <CommandEmpty>No models found</CommandEmpty>
              <CommandGroup>
                {models?.map((model) => (
                  <CommandItem
                    key={model.title}
                    value={model.title}
                    onSelect={() => handleSelect(model.title)}
                    className="text-xs"
                  >
                    <Check size={14} className={cn("mr-1 flex-shrink-0", model.title === currentModel ? "opacity-100" : "opacity-0")} />
                    <span className="truncate flex-1">{model.title}</span>
                    {model.hash && (
                      <span className="text-3xs text-muted-foreground font-mono pl-2">{model.hash.slice(0, 8)}</span>
                    )}
                  </CommandItem>
                ))}
              </CommandGroup>
            </CommandList>
          </Command>
        </PopoverContent>
      </Popover>

      {pipelineClass && (
        <span className="text-3xs text-muted-foreground whitespace-nowrap">{pipelineClass}</span>
      )}

      <Button variant="ghost" size="icon-sm" title="Reload model" disabled={isModelLoading} onClick={() => reloadModel.mutate(undefined)}>
        <RefreshCw size={14} />
      </Button>
      <Button variant="ghost" size="icon-sm" title="Unload model" disabled={isModelLoading} onClick={() => unloadModel.mutate(undefined)}>
        <ArrowBigDownDash size={14} />
      </Button>
      <Button variant="ghost" size="icon-sm" title="Refresh model list" disabled={isModelLoading} onClick={() => refreshModels.mutate(undefined)}>
        <FolderSync size={14} />
      </Button>
    </div>
  );
}
