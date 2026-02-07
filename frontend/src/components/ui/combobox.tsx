import { useState } from "react";
import { Check, ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";
import { Popover, PopoverContent, PopoverTrigger } from "./popover";
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from "./command";

export type ComboboxOption = string | { value: string; label: string };

interface ComboboxProps {
  value: string;
  onValueChange: (value: string) => void;
  options: readonly ComboboxOption[];
  placeholder?: string;
  searchPlaceholder?: string;
  emptyText?: string;
  className?: string;
  align?: "start" | "center" | "end";
}

export function Combobox({
  value,
  onValueChange,
  options,
  placeholder = "Select...",
  searchPlaceholder = "Search...",
  emptyText = "No results",
  className,
  align = "start",
}: ComboboxProps) {
  const [open, setOpen] = useState(false);

  const currentLabel = options.reduce<string | undefined>((found, opt) => {
    if (found) return found;
    if (typeof opt === "string") return opt === value ? opt : undefined;
    return opt.value === value ? opt.label : undefined;
  }, undefined) ?? value;

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <button
          type="button"
          className={cn(
            "border-input bg-transparent flex w-fit items-center justify-between gap-2 rounded-md border px-3 py-2 text-sm whitespace-nowrap shadow-xs transition-[color,box-shadow] outline-none focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px] disabled:cursor-not-allowed disabled:opacity-50",
            className,
          )}
        >
          <span className="truncate">{value ? currentLabel : placeholder}</span>
          <ChevronDown size={12} className="shrink-0 opacity-50" />
        </button>
      </PopoverTrigger>
      <PopoverContent
        className="p-0 w-[--radix-popover-trigger-width] min-w-[--radix-popover-trigger-width]"
        align={align}
      >
        <Command>
          {options.length > 6 && <CommandInput placeholder={searchPlaceholder} />}
          <CommandList>
            <CommandEmpty>{emptyText}</CommandEmpty>
            <CommandGroup>
              {options.map((opt) => {
                const v = typeof opt === "string" ? opt : opt.value;
                const l = typeof opt === "string" ? opt : opt.label;
                return (
                  <CommandItem
                    key={v}
                    value={v}
                    keywords={typeof opt === "string" ? undefined : [l]}
                    onSelect={() => { onValueChange(v); setOpen(false); }}
                    className="text-xs"
                  >
                    <Check size={14} className={cn("shrink-0", v === value ? "opacity-100" : "opacity-0")} />
                    {l}
                  </CommandItem>
                );
              })}
            </CommandGroup>
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  );
}
