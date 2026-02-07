import { useState, useCallback } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";

interface ParamSectionProps {
  title: string;
  defaultOpen?: boolean;
  children: React.ReactNode;
}

export function ParamSection({ title, defaultOpen = true, children }: ParamSectionProps) {
  const [open, setOpen] = useState(defaultOpen);
  const toggle = useCallback(() => setOpen((o) => !o), []);

  return (
    <div className="mb-3">
      <button
        type="button"
        onClick={toggle}
        className="flex items-center gap-1 text-[11px] font-medium text-muted-foreground hover:text-foreground mb-1.5 uppercase tracking-wider"
      >
        {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        {title}
      </button>
      {open && <div className="flex flex-col gap-2">{children}</div>}
    </div>
  );
}
