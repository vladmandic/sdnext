import { useUiStore } from "@/stores/uiStore";
import { GalleryMetadata } from "@/components/gallery/GalleryMetadata";

export function RightPanel() {
  const activeView = useUiStore((s) => s.activeSidebarView);

  if (activeView === "gallery") {
    return <GalleryMetadata />;
  }

  return (
    <div className="flex flex-col h-full p-3 text-muted-foreground text-xs">
      <span className="text-[11px] font-medium uppercase tracking-wider mb-2">Info</span>
      <p>Output details and metadata will appear here.</p>
    </div>
  );
}
