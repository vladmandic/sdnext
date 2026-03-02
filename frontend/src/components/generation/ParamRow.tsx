import { ParamLabel } from "./ParamLabel";

interface ParamRowProps {
  label: string;
  children: React.ReactNode;
  disabled?: boolean;
  tooltip?: string;
}

/** Label stacked above its control. Use inside ParamGrid for 2-column pairs. */
export function ParamRow({ label, children, disabled, tooltip }: ParamRowProps) {
  return (
    <div className={disabled ? "opacity-50 pointer-events-none" : undefined}>
      <ParamLabel className="text-2xs text-muted-foreground mb-0.5 block" tooltip={tooltip}>{label}</ParamLabel>
      {children}
    </div>
  );
}

/** 2-column grid for pairing related ParamRow items side-by-side. */
export function ParamGrid({ children }: { children: React.ReactNode }) {
  return <div className="grid grid-cols-2 gap-x-2 gap-y-2">{children}</div>;
}
