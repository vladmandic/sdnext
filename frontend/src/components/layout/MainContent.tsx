import { useUiStore } from "@/stores/uiStore";
import { CanvasView } from "@/components/generation/CanvasView";
import { GalleryView } from "@/components/gallery/GalleryView";
import { CaptionView } from "@/components/caption/CaptionView";
import { ProcessView } from "@/components/process/ProcessView";
import { VideoView } from "@/components/video/VideoView";

export function MainContent() {
  const activeView = useUiStore((s) => s.activeSidebarView);

  switch (activeView) {
    case "images":
      return <CanvasView />;
    case "gallery":
      return <GalleryView />;
    case "video":
      return <VideoView />;
    case "process":
      return <ProcessView />;
    case "caption":
      return <CaptionView />;
    default:
      return <CanvasView />;
  }
}
