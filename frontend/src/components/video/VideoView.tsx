import { Download } from "lucide-react";
import { useVideoStore } from "@/stores/videoStore";
import { VideoPlayer } from "@/components/video/VideoPlayer";
import { Button } from "@/components/ui/button";

export function VideoView() {
  const resultVideoUrl = useVideoStore((s) => s.resultVideoUrl);

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
    </div>
  );
}
