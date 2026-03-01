import { useUiStore } from "@/stores/uiStore";
import { useCapabilities } from "@/api/hooks/useServer";
import { NAV_ITEMS } from "@/lib/constants";
import { CanvasView } from "@/components/generation/CanvasView";
import { GalleryView } from "@/components/gallery/GalleryView";
import { CaptionView } from "@/components/caption/CaptionView";
import { ProcessView } from "@/components/process/ProcessView";
import { VideoView } from "@/components/video/VideoView";
import { VideoOff } from "lucide-react";

function UnavailablePlaceholder({ label }: { label: string }) {
  return (
    <div className="flex flex-col items-center justify-center h-full gap-3 text-muted-foreground">
      <VideoOff size={48} strokeWidth={1} />
      <p className="text-sm">{label} requires additional models to be installed.</p>
    </div>
  );
}

export function MainContent() {
  const activeView = useUiStore((s) => s.activeSidebarView);
  const capabilities = useCapabilities();

  const navItem = NAV_ITEMS.find((n) => n.id === activeView);
  if (navItem?.capability && capabilities && !capabilities[navItem.capability]) {
    return <UnavailablePlaceholder label={navItem.label} />;
  }

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
