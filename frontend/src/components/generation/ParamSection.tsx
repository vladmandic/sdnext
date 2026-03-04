import { useState, useCallback, useEffect } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";

interface ParamSectionProps {
  title: string;
  defaultOpen?: boolean;
  action?: React.ReactNode;
  children: React.ReactNode;
}

export function ParamSection({ title, defaultOpen = true, action, children }: ParamSectionProps) {
  const [open, setOpen] = useState(defaultOpen);
  const toggle = useCallback(() => setOpen((o) => !o), []);

  useEffect(() => {
    const handler = (e: Event) => {
      const detail = (e as CustomEvent<{ section: string }>).detail;
      if (detail.section === title.toLowerCase()) setOpen(true);
    };
    document.addEventListener("param-section-expand", handler);
    return () => document.removeEventListener("param-section-expand", handler);
  }, [title]);

  return (
    <div data-section={title.toLowerCase()} className="mb-3">
      <div className="flex items-center mb-1.5">
        <button
          type="button"
          onClick={toggle}
          className="flex items-center gap-1 text-2xs font-medium text-muted-foreground hover:text-foreground uppercase tracking-wider"
        >
          {title}
          {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        </button>
        {action && <div className="ml-auto">{action}</div>}
      </div>
      {open && <div className="flex flex-col gap-2">{children}</div>}
    </div>
  );
}
