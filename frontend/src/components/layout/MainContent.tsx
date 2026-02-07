import { useUiStore } from "@/stores/uiStore";
import { GenerateView } from "@/components/generation/GenerateView";
import { GalleryView } from "@/components/gallery/GalleryView";
import { CaptionView } from "@/components/caption/CaptionView";

export function MainContent() {
  const activeView = useUiStore((s) => s.activeSidebarView);

  switch (activeView) {
    case "images":
      return <GenerateView />;
    case "gallery":
      return <GalleryView />;
    case "video":
      return <PlaceholderView title="Video" description="Video generation" />;
    case "process":
      return <PlaceholderView title="Process" description="Post-processing and upscaling" />;
    case "caption":
      return <CaptionView />;
    default:
      return <GenerateView />;
  }
}

function PlaceholderView({ title, description }: { title: string; description: string }) {
  return (
    <div className="flex flex-col items-center justify-center h-full text-muted-foreground">
      <h2 className="text-xl font-medium text-foreground">{title}</h2>
      <p className="text-sm mt-1">{description}</p>
      <p className="text-xs mt-4 opacity-50">Coming soon</p>
    </div>
  );
}
