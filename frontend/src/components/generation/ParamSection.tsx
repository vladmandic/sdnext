import { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";

interface ParamSectionProps {
  title: string;
  defaultOpen?: boolean;
  children: React.ReactNode;
}

export function ParamSection({ title, defaultOpen = true, children }: ParamSectionProps) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <Collapsible open={open} onOpenChange={setOpen} className="mb-3">
      <CollapsibleTrigger className="flex items-center gap-1 text-[11px] font-medium text-muted-foreground hover:text-foreground mb-1.5 uppercase tracking-wider">
        {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        {title}
      </CollapsibleTrigger>
      <CollapsibleContent>
        <div className="flex flex-col gap-2">{children}</div>
      </CollapsibleContent>
    </Collapsible>
  );
}
