import { useServerInfo } from "@/api/hooks/useServer";
import { Wifi, WifiOff, Loader2 } from "lucide-react";

export function ConnectionIndicator() {
  const { data: serverInfo, isLoading, isError } = useServerInfo();

  if (isLoading) {
    return (
      <div className="flex items-center gap-1.5 text-muted-foreground text-xs px-2">
        <Loader2 size={14} className="animate-spin" />
        <span>Connecting...</span>
      </div>
    );
  }

  if (isError || !serverInfo) {
    return (
      <div className="flex items-center gap-1.5 text-destructive text-xs px-2">
        <WifiOff size={14} />
        <span>Disconnected</span>
      </div>
    );
  }

  const label = serverInfo.platform || serverInfo.backend || "Connected";

  return (
    <div className="flex items-center gap-1.5 text-emerald-500 text-xs px-2">
      <Wifi size={14} />
      <span className="hidden sm:inline">{label}</span>
    </div>
  );
}
