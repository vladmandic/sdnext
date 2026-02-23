import { useState } from "react";
import { RotateCcw, PowerOff, Activity } from "lucide-react";
import { useRestartServer, useShutdownServer, useToggleProfiling } from "@/api/hooks/useSystem";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { cn } from "@/lib/utils";

import { OverviewSubTab } from "@/components/system/sub-tabs/OverviewSubTab";
import { UpdateSubTab } from "@/components/system/sub-tabs/UpdateSubTab";
import { HistorySubTab } from "@/components/system/sub-tabs/HistorySubTab";
import { GpuMonitorSubTab } from "@/components/system/sub-tabs/GpuMonitorSubTab";
import { SystemInfoSubTab } from "@/components/system/sub-tabs/SystemInfoSubTab";
import { BenchmarkSubTab } from "@/components/system/sub-tabs/BenchmarkSubTab";
import { StorageSubTab } from "@/components/system/sub-tabs/StorageSubTab";

const SUB_TABS = [
  "Overview",
  "Storage",
  "Update",
  "History",
  "GPU Monitor",
  "System Info",
  "Benchmark",
] as const;

type SubTab = (typeof SUB_TABS)[number];

export function SystemTab() {
  const [active, setActive] = useState<SubTab>("Overview");
  const [confirmAction, setConfirmAction] = useState<"restart" | "shutdown" | null>(null);
  const [profiling, setProfiling] = useState(false);

  const restartServer = useRestartServer();
  const shutdownServer = useShutdownServer();
  const toggleProfiling = useToggleProfiling();

  function handleConfirm() {
    if (confirmAction === "restart") restartServer.mutate();
    else if (confirmAction === "shutdown") shutdownServer.mutate();
    setConfirmAction(null);
  }

  function handleProfiling() {
    toggleProfiling.mutate(undefined, {
      onSuccess: (data) => {
        if (data && typeof data === "object" && "enabled" in data) {
          setProfiling(data.enabled as boolean);
        }
      },
    });
  }

  return (
    <div>
      <div className="sticky top-0 z-10 bg-card p-2 space-y-2 border-b border-border">
        <div className="flex items-center gap-1 flex-wrap">
          <Button size="sm" variant="destructive-soft" onClick={() => setConfirmAction("restart")}>
            <RotateCcw />
            Restart
          </Button>
          <Button size="sm" variant="destructive-soft" onClick={() => setConfirmAction("shutdown")}>
            <PowerOff />
            Shutdown
          </Button>
          <Button
            size="sm"
            variant={profiling ? "default" : "ghost"}
            className="h-7 px-2 text-2xs ml-auto"
            title="Toggle profiling"
            onClick={handleProfiling}
          >
            <Activity className="h-3.5 w-3.5 mr-1" />
            {profiling ? "Stop" : "Profile"}
          </Button>
        </div>

        <div className="flex items-center gap-1 flex-wrap">
          {SUB_TABS.map((tab) => (
            <button
              key={tab}
              type="button"
              onClick={() => setActive(tab)}
              className={cn(
                "px-2 py-0.5 rounded-full text-2xs font-medium transition-colors",
                active === tab
                  ? "bg-accent text-accent-foreground"
                  : "bg-muted text-muted-foreground hover:text-foreground",
              )}
            >
              {tab}
            </button>
          ))}
        </div>
      </div>

      <div className="p-3">
        {active === "Overview" && <OverviewSubTab />}
        {active === "Storage" && <StorageSubTab />}
        {active === "Update" && <UpdateSubTab />}
        {active === "History" && <HistorySubTab />}
        {active === "GPU Monitor" && <GpuMonitorSubTab />}
        {active === "System Info" && <SystemInfoSubTab />}
        {active === "Benchmark" && <BenchmarkSubTab />}
      </div>

      <Dialog open={confirmAction !== null} onOpenChange={(open) => !open && setConfirmAction(null)}>
        <DialogContent showCloseButton={false} className="sm:max-w-sm">
          <DialogHeader>
            <DialogTitle>
              {confirmAction === "restart" ? "Restart Server" : "Shutdown Server"}
            </DialogTitle>
            <DialogDescription>
              {confirmAction === "restart"
                ? "The server will restart. You may lose connection temporarily."
                : "The server will shut down completely. You will need to start it manually."}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" size="sm" onClick={() => setConfirmAction(null)}>Cancel</Button>
            <Button variant="destructive" size="sm" onClick={handleConfirm}>
              {confirmAction === "restart" ? "Restart" : "Shutdown"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
