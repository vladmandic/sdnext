import { useExtensions } from "@/api/hooks/useExtensions";
import { Badge } from "@/components/ui/badge";

export function ExtensionsTab() {
  const { data: extensions, isLoading } = useExtensions();

  if (isLoading) {
    return <div className="p-3 text-xs text-muted-foreground">Loading extensions...</div>;
  }

  return (
    <div className="p-3 space-y-1">
      {extensions?.map((ext) => (
        <div key={ext.name} className="flex items-center justify-between py-1.5 px-2 rounded hover:bg-muted/50">
          <span className="text-xs truncate min-w-0">{ext.name}</span>
          <div className="flex items-center gap-1.5 shrink-0">
            {ext.version && <span className="text-3xs text-muted-foreground">{ext.version}</span>}
            <Badge variant={ext.enabled ? "default" : "secondary"} className="text-3xs px-1.5 py-0">
              {ext.enabled ? "on" : "off"}
            </Badge>
          </div>
        </div>
      ))}
      {(!extensions || extensions.length === 0) && (
        <p className="text-xs text-muted-foreground">No extensions found.</p>
      )}
    </div>
  );
}
