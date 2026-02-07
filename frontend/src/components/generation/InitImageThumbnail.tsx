import { useCallback, useRef } from "react";
import { useImg2ImgStore } from "@/stores/img2imgStore";
import { X } from "lucide-react";
import { Button } from "@/components/ui/button";

export function InitImageThumbnail() {
  const initImageData = useImg2ImgStore((s) => s.initImageData);
  const initImageName = useImg2ImgStore((s) => s.initImageName);
  const initImageWidth = useImg2ImgStore((s) => s.initImageWidth);
  const initImageHeight = useImg2ImgStore((s) => s.initImageHeight);
  const setInitImage = useImg2ImgStore((s) => s.setInitImage);
  const clearInitImage = useImg2ImgStore((s) => s.clearInitImage);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && file.type.startsWith("image/")) {
      setInitImage(file);
    }
    e.target.value = "";
  }, [setInitImage]);

  const openPicker = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  if (!initImageData) return null;

  const truncatedName = initImageName.length > 20
    ? `${initImageName.slice(0, 17)}...`
    : initImageName;

  return (
    <>
      <div className="flex items-center gap-2 h-10">
        {/* Thumbnail */}
        <button
          type="button"
          onClick={openPicker}
          className="flex-shrink-0 w-10 h-10 rounded overflow-hidden border border-border hover:border-foreground/30 transition-colors cursor-pointer"
          title="Click to replace image"
        >
          <img
            src={initImageData}
            alt="Init"
            className="w-full h-full object-cover"
          />
        </button>

        {/* Info */}
        <div className="flex-1 min-w-0">
          <p className="text-[11px] truncate" title={initImageName}>
            {truncatedName}
          </p>
          <p className="text-[10px] text-muted-foreground">
            {initImageWidth}&times;{initImageHeight}
          </p>
        </div>

        {/* Clear */}
        <Button
          variant="ghost"
          size="icon-xs"
          onClick={clearInitImage}
          className="text-muted-foreground flex-shrink-0"
          title="Clear image"
        >
          <X size={12} />
        </Button>
      </div>
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={handleFileInput}
        className="hidden"
      />
    </>
  );
}
