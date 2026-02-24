import { useState, type ReactNode } from "react";
import { Check, ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";
import { Popover, PopoverContent, PopoverTrigger } from "./popover";
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from "./command";

export type ComboboxOption = string | { value: string; label: string };

export interface ComboboxGroup {
  heading: string;
  options: readonly ComboboxOption[];
}

interface ComboboxBaseProps {
  value: string;
  onValueChange: (value: string) => void;
  placeholder?: string;
  searchPlaceholder?: string;
  emptyText?: string;
  className?: string;
  align?: "start" | "center" | "end";
  renderLabel?: (value: string, label: string) => ReactNode;
}

interface ComboboxFlatProps extends ComboboxBaseProps {
  options: readonly ComboboxOption[];
  groups?: undefined;
}

interface ComboboxGroupedProps extends ComboboxBaseProps {
  groups: readonly ComboboxGroup[];
  options?: undefined;
}

type ComboboxProps = ComboboxFlatProps | ComboboxGroupedProps;

function getOptionValue(opt: ComboboxOption): string {
  return typeof opt === "string" ? opt : opt.value;
}

function getOptionLabel(opt: ComboboxOption): string {
  return typeof opt === "string" ? opt : opt.label;
}

export function Combobox({
  value,
  onValueChange,
  options,
  groups,
  placeholder = "Select...",
  searchPlaceholder = "Search...",
  emptyText = "No results",
  className,
  align = "start",
  renderLabel,
}: ComboboxProps) {
  const [open, setOpen] = useState(false);

  const allOptions: readonly ComboboxOption[] = groups
    ? groups.flatMap((g) => g.options)
    : (options ?? []);

  const currentLabel = allOptions.reduce<string | undefined>((found, opt) => {
    if (found) return found;
    return getOptionValue(opt) === value ? getOptionLabel(opt) : undefined;
  }, undefined) ?? value;

  const renderItem = (opt: ComboboxOption) => {
    const v = getOptionValue(opt);
    const l = getOptionLabel(opt);
    return (
      <CommandItem
        key={v}
        value={v}
        keywords={typeof opt === "string" ? undefined : [l]}
        onSelect={() => { onValueChange(v); setOpen(false); }}
        className="text-2xs"
      >
        <Check size={14} className={cn("shrink-0", v === value ? "opacity-100" : "opacity-0")} />
        {renderLabel ? renderLabel(v, l) : l}
      </CommandItem>
    );
  };

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <button
          type="button"
          className={cn(
            "border-input bg-transparent flex w-full items-center justify-between gap-2 rounded-md border px-3 py-2 text-2xs shadow-xs transition-[color,box-shadow] outline-none focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px] disabled:cursor-not-allowed disabled:opacity-50",
            className,
          )}
        >
          <span className="break-words text-left">{value ? (renderLabel ? renderLabel(value, currentLabel) : currentLabel) : placeholder}</span>
          <ChevronDown size={12} className="shrink-0 opacity-50" />
        </button>
      </PopoverTrigger>
      <PopoverContent
        className="p-0 w-[--radix-popover-trigger-width] min-w-[--radix-popover-trigger-width]"
        align={align}
      >
        <Command>
          {allOptions.length > 6 && <CommandInput placeholder={searchPlaceholder} />}
          <CommandList>
            <CommandEmpty>{emptyText}</CommandEmpty>
            {groups
              ? groups.map((g) => (
                <CommandGroup
                  key={g.heading}
                  heading={g.heading}
                  className="overflow-visible [&_[cmdk-group-heading]]:bg-popover [&_[cmdk-group-heading]]:sticky [&_[cmdk-group-heading]]:top-0 [&_[cmdk-group-heading]]:z-10"
                >
                  {g.options.map(renderItem)}
                </CommandGroup>
              ))
              : (
                <CommandGroup>
                  {(options ?? []).map(renderItem)}
                </CommandGroup>
              )}
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  );
}
