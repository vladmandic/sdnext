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
  if (!help) return <Label className={className}>{children}</Label>;
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Label className={cn(className, "cursor-help")}>{children}</Label>
      </TooltipTrigger>
      <TooltipContent className="max-w-xs text-xs">
        <span dangerouslySetInnerHTML={{ __html: help }} />
      </TooltipContent>
    </Tooltip>
  );
}
