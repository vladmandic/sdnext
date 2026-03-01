import { useUiStore } from "@/stores/uiStore";
import { ActionBar } from "@/components/generation/ActionBar";
import { ResultGallery } from "@/components/generation/ResultGallery";
import { PromptsTab } from "@/components/generation/tabs/PromptsTab";
import { SamplerTab } from "@/components/generation/tabs/SamplerTab";
import { GuidanceTab } from "@/components/generation/tabs/GuidanceTab";
import { RefineTab } from "@/components/generation/tabs/RefineTab";
import { AdvancedTab } from "@/components/generation/tabs/AdvancedTab";
import { ColorTab } from "@/components/generation/tabs/ColorTab";
import { DetailTab } from "@/components/generation/tabs/DetailTab";
import { ControlTab } from "@/components/generation/tabs/ControlTab";
import { ScriptsTab } from "@/components/generation/tabs/ScriptsTab";
import { CaptionPanel } from "@/components/caption/CaptionPanel";
import { GalleryPanel } from "@/components/gallery/GalleryPanel";
import { ProcessPanel } from "@/components/process/ProcessPanel";
import { VideoPanel } from "@/components/video/VideoPanel";
import { ScrollArea } from "@/components/ui/scroll-area";

export function LeftPanel() {
  const activeView = useUiStore((s) => s.activeSidebarView);
  const activeSubTab = useUiStore((s) => s.activeImagesSubTab);

  if (activeView === "caption") {
    return <CaptionPanel />;
  }

  if (activeView === "gallery") {
    return <GalleryPanel />;
  }

  if (activeView === "process") {
    return <ProcessPanel />;
  }

  if (activeView === "video") {
    return <VideoPanel />;
  }

  if (activeView === "images") {
    return (
      <div className="flex flex-col h-full min-w-0">
        {/* Action bar */}
        <div className="px-3 py-2 border-b border-border">
          <ActionBar />
        </div>

        {/* Sub-tab content */}
        <ScrollArea className="flex-1 min-h-0">
          <div className="p-3 min-w-0">
            <ImagesSubTabContent activeSubTab={activeSubTab} />
          </div>
        </ScrollArea>

        {/* Result thumbnails */}
        <div className="border-t border-border px-2 py-1.5">
          <ResultGallery />
        </div>
      </div>
    );
  }

  return (
    <div className="flex items-center justify-center h-full text-muted-foreground text-sm">
      Panel for {activeView}
    </div>
  );
}

function ImagesSubTabContent({ activeSubTab }: { activeSubTab: string }) {
  switch (activeSubTab) {
    case "prompts":
      return <PromptsTab />;
    case "sampler":
      return <SamplerTab />;
    case "guidance":
      return <GuidanceTab />;
    case "refine":
      return <RefineTab />;
    case "detail":
      return <DetailTab />;
    case "advanced":
      return <AdvancedTab />;
    case "color":
      return <ColorTab />;
    case "control":
      return <ControlTab />;
    case "scripts":
      return <ScriptsTab />;
    default:
      return <PromptsTab />;
  }
}
