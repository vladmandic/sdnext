import { useUiStore } from "@/stores/uiStore";
import { useImg2ImgStore } from "@/stores/img2imgStore";
import { ActionBar } from "@/components/generation/ActionBar";
import { ModeSelector } from "@/components/generation/ModeSelector";
import { InitImageThumbnail } from "@/components/generation/InitImageThumbnail";
import { MaskParams } from "@/components/generation/MaskParams";
import { ResultGallery } from "@/components/generation/ResultGallery";
import { PromptsTab } from "@/components/generation/tabs/PromptsTab";
import { SamplerTab } from "@/components/generation/tabs/SamplerTab";
import { GuidanceTab } from "@/components/generation/tabs/GuidanceTab";
import { RefineTab } from "@/components/generation/tabs/RefineTab";
import { AdvancedTab } from "@/components/generation/tabs/AdvancedTab";
import { DetailTab } from "@/components/generation/tabs/DetailTab";
import { AdaptersTab } from "@/components/generation/tabs/AdaptersTab";
import { ControlTab } from "@/components/generation/tabs/ControlTab";
import { ScriptsTab } from "@/components/generation/tabs/ScriptsTab";
import { CaptionPanel } from "@/components/caption/CaptionPanel";
import { GalleryPanel } from "@/components/gallery/GalleryPanel";
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

  const generationMode = useUiStore((s) => s.generationMode);
  const hasInitImage = useImg2ImgStore((s) => s.initImageData !== null);

  if (activeView === "images") {
    return (
      <div className="flex flex-col h-full">
        {/* Mode selector */}
        <div className="px-3 pt-2">
          <ModeSelector />
        </div>

        {/* Action bar */}
        <div className="px-3 py-2 border-b border-border">
          <ActionBar />
        </div>

        {/* Init image thumbnail + mask params (img2img with image loaded) */}
        {generationMode === "img2img" && hasInitImage && (
          <div className="px-3 py-1.5 border-b border-border">
            <InitImageThumbnail />
            <MaskParams />
          </div>
        )}

        {/* Sub-tab content */}
        <ScrollArea className="flex-1">
          <div className="p-3">
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
    case "adapters":
      return <AdaptersTab />;
    case "control":
      return <ControlTab />;
    case "scripts":
      return <ScriptsTab />;
    default:
      return <PromptsTab />;
  }
}
