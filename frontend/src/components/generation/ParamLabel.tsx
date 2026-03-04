import { useEffect, useRef, useState } from "react";
import { Label } from "@/components/ui/label";
import { Tooltip, TooltipTrigger, TooltipContent } from "@/components/ui/tooltip";
import { getParamHelp } from "@/data/parameterHelp";
import { cn } from "@/lib/utils";

interface ParamLabelProps {
  children: string;
  className?: string;
  tooltip?: string;
}

export function ParamLabel({ children, className, tooltip }: ParamLabelProps) {
  const help = tooltip ?? getParamHelp(children);
  const [pinned, setPinned] = useState(false);
  const [hovered, setHovered] = useState(false);
  const triggerRef = useRef<HTMLLabelElement>(null);
  const hoverTimer = useRef<ReturnType<typeof setTimeout>>(undefined);

  useEffect(() => () => clearTimeout(hoverTimer.current), []);

  useEffect(() => {
    if (!pinned) return;
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setPinned(false);
    };
    const handleClick = (e: PointerEvent) => {
      const target = e.target as Node;
      if (triggerRef.current?.contains(target)) return;
      const content = document.querySelector("[data-slot='tooltip-content']");
      if (content?.contains(target)) return;
      setPinned(false);
    };
    document.addEventListener("keydown", handleKey);
    document.addEventListener("pointerdown", handleClick);
    return () => {
      document.removeEventListener("keydown", handleKey);
      document.removeEventListener("pointerdown", handleClick);
    };
  }, [pinned]);

  if (!help) return <Label className={className}>{children}</Label>;

  return (
    <Tooltip open={hovered || pinned}>
      <TooltipTrigger asChild>
        <Label
          ref={triggerRef}
          className={cn(className, "cursor-help", pinned && "underline decoration-dotted underline-offset-2 decoration-muted-foreground")}
          onPointerEnter={() => { hoverTimer.current = setTimeout(() => setHovered(true), 300); }}
          onPointerLeave={() => { clearTimeout(hoverTimer.current); setHovered(false); }}
          onClick={(e) => { e.preventDefault(); setPinned((p) => !p); }}
        >
          {children}
        </Label>
      </TooltipTrigger>
      <TooltipContent>
        <span dangerouslySetInnerHTML={{ __html: help }} />
      </TooltipContent>
    </Tooltip>
  );
}
