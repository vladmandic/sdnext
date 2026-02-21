import { Download } from "lucide-react";
import { useVideoStore } from "@/stores/videoStore";
import { useJobQueueStore, selectDomainActive, selectDomainProgress } from "@/stores/jobStore";
import { VideoPlayer } from "@/components/video/VideoPlayer";
import { Button } from "@/components/ui/button";

export function VideoView() {
  const resultVideoUrl = useVideoStore((s) => s.resultVideoUrl);
  const isVideoActive = useJobQueueStore(selectDomainActive("video"));
  const isFramepackActive = useJobQueueStore(selectDomainActive("framepack"));
  const isLtxActive = useJobQueueStore(selectDomainActive("ltx"));
  const isGenerating = isVideoActive || isFramepackActive || isLtxActive;
  const videoProgress = useJobQueueStore(selectDomainProgress("video"));
  const fpProgress = useJobQueueStore(selectDomainProgress("framepack"));
  const ltxProgress = useJobQueueStore(selectDomainProgress("ltx"));
  const progress = Math.max(videoProgress, fpProgress, ltxProgress);

  const progressPct = Math.round(progress * 100);

  return (
    <div className="h-full relative">
      <VideoPlayer src={resultVideoUrl} />
      {resultVideoUrl && (
        <a href={resultVideoUrl} download className="absolute top-4 right-4">
          <Button variant="secondary" size="icon-sm">
            <Download size={14} />
          </Button>
        </a>
      )}
      {isGenerating && (
        <div className="absolute inset-x-0 bottom-0 p-4">
          <div className="flex items-center gap-2 bg-background/80 backdrop-blur-sm rounded-lg px-3 py-2">
            <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
              <div
                className="h-full bg-primary rounded-full transition-[width] duration-300"
                style={{ width: `${progressPct}%` }}
              />
            </div>
            <span className="text-xs text-muted-foreground tabular-nums min-w-[3ch]">{progressPct}%</span>
          </div>
        </div>
      )}
    </div>
  );
}
