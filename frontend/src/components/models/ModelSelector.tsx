import { useModelList, useLoadModel } from "@/api/hooks/useModels";
import { useOptions } from "@/api/hooks/useSettings";
import { useModelStore } from "@/stores/modelStore";
import { RefreshCw, Check, ChevronsUpDown } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from "@/components/ui/command";

export function ModelSelector() {
  const { data: models } = useModelList();
  const { data: options } = useOptions();
  const loadModel = useLoadModel();
  const isModelLoading = useModelStore((s) => s.isModelLoading);
  const setModelLoading = useModelStore((s) => s.setModelLoading);

  const [open, setOpen] = useState(false);

  const currentModel = (options?.sd_model_checkpoint as string) ?? "No model loaded";

  async function handleSelect(title: string) {
    if (title === currentModel) {
      setOpen(false);
      return;
    }
    setOpen(false);
    setModelLoading(true);
    try {
      await loadModel.mutateAsync(title);
      toast.success("Model loaded", { description: title });
    } catch (err) {
      toast.error("Failed to load model", { description: err instanceof Error ? err.message : String(err) });
    } finally {
      setModelLoading(false);
    }
  }

  return (
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
      <PopoverContent className="w-[400px] p-0" align="start">
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
                    <span className="text-[10px] text-muted-foreground font-mono pl-2">{model.hash.slice(0, 8)}</span>
                  )}
                </CommandItem>
              ))}
            </CommandGroup>
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  );
}
