import { useState } from "react";
import { X } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from "@/components/ui/command";

interface XyzValueTagsProps {
  choices: string[];
  selected: string[];
  onChange: (selected: string[]) => void;
}

export function XyzValueTags({ choices, selected, onChange }: XyzValueTagsProps) {
  const [open, setOpen] = useState(false);

  const toggle = (value: string) => {
    if (selected.includes(value)) {
      onChange(selected.filter((s) => s !== value));
    } else {
      onChange([...selected, value]);
    }
  };

  const remove = (value: string) => {
    onChange(selected.filter((s) => s !== value));
  };

  return (
    <div className="space-y-1.5">
      {selected.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {selected.map((v) => (
            <Badge key={v} variant="secondary" className="text-3xs gap-0.5 pr-0.5">
              {v}
              <button type="button" className="ml-0.5 rounded-full p-0.5 hover:bg-muted" onClick={() => remove(v)}>
                <X size={10} />
              </button>
            </Badge>
          ))}
        </div>
      )}
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <button
            type="button"
            className="border-input bg-transparent flex w-full items-center rounded-md border px-3 py-1.5 text-2xs shadow-xs transition-[color,box-shadow] outline-none focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px] text-muted-foreground"
          >
            {selected.length === 0 ? "Click to select values..." : `${selected.length} selected — click to add more`}
          </button>
        </PopoverTrigger>
        <PopoverContent className="p-0 w-[--radix-popover-trigger-width] min-w-[--radix-popover-trigger-width]" align="start">
          <Command>
            {choices.length > 6 && <CommandInput placeholder="Search..." />}
            <CommandList>
              <CommandEmpty>No results</CommandEmpty>
              <CommandGroup>
                {choices.map((c) => (
                  <CommandItem
                    key={c}
                    value={c}
                    onSelect={() => toggle(c)}
                    className="text-2xs"
                  >
                    <span className={selected.includes(c) ? "font-semibold text-primary" : ""}>{c}</span>
                    {selected.includes(c) && <span className="ml-auto text-3xs text-muted-foreground">selected</span>}
                  </CommandItem>
                ))}
              </CommandGroup>
            </CommandList>
          </Command>
        </PopoverContent>
      </Popover>
    </div>
  );
}
