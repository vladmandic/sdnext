import { useState, useCallback } from "react";
import { MoreVertical, ImagePlus, Scissors, ArrowUpFromLine, RotateCcw, FastForward } from "lucide-react";
import { toast } from "sonner";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuSeparator, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import { Button } from "@/components/ui/button";
import { FramePickerDialog } from "./FramePickerDialog";
import { extractFrameFromVideo, sendFrameToVideoInit, sendFrameToUpscale, restoreVideoSettings } from "@/lib/sendTo";
import type { VideoResult } from "@/api/types/video";

interface VideoResultActionsProps {
  result: VideoResult;
}

export function VideoResultActions({ result }: VideoResultActionsProps) {
  const [framePickerOpen, setFramePickerOpen] = useState(false);

  const handleSendFirstFrame = useCallback(async () => {
    try {
      const blob = await extractFrameFromVideo(result.videoUrl, 0);
      sendFrameToVideoInit(blob);
      toast.success("First frame sent to Init Image");
    } catch {
      toast.error("Failed to extract frame");
    }
  }, [result.videoUrl]);

  const handleSendLastFrame = useCallback(async () => {
    try {
      const blob = await extractFrameFromVideo(result.videoUrl, 999999);
      sendFrameToVideoInit(blob);
      toast.success("Last frame sent to Init Image");
    } catch {
      toast.error("Failed to extract frame");
    }
  }, [result.videoUrl]);

  const handleFrameCapture = useCallback((blob: Blob) => {
    sendFrameToVideoInit(blob);
    toast.success("Captured frame sent to Init Image");
  }, []);

  const handleSendToUpscale = useCallback(async () => {
    try {
      const blob = await extractFrameFromVideo(result.videoUrl, 0);
      sendFrameToUpscale(blob);
      toast.success("Frame sent to Upscale");
    } catch {
      toast.error("Failed to extract frame");
    }
  }, [result.videoUrl]);

  const handleReuseSettings = useCallback(() => {
    restoreVideoSettings(result.params);
    toast.success("Video settings restored");
  }, [result.params]);

  const handleExtend = useCallback(async () => {
    try {
      const blob = await extractFrameFromVideo(result.videoUrl, 999999);
      sendFrameToVideoInit(blob);
      restoreVideoSettings(result.params);
      toast.success("Ready to extend video");
    } catch {
      toast.error("Failed to extract frame");
    }
  }, [result.videoUrl, result.params]);

  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="secondary" size="icon-sm">
            <MoreVertical size={14} />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end">
          <DropdownMenuItem onClick={handleSendFirstFrame}>
            <ImagePlus size={14} />
            <span>Send first frame to Init</span>
          </DropdownMenuItem>
          <DropdownMenuItem onClick={handleSendLastFrame}>
            <ImagePlus size={14} />
            <span>Send last frame to Init</span>
          </DropdownMenuItem>
          <DropdownMenuItem onClick={() => setFramePickerOpen(true)}>
            <Scissors size={14} />
            <span>Extract frame...</span>
          </DropdownMenuItem>
          <DropdownMenuSeparator />
          <DropdownMenuItem onClick={handleSendToUpscale}>
            <ArrowUpFromLine size={14} />
            <span>Send frame to Upscale</span>
          </DropdownMenuItem>
          <DropdownMenuSeparator />
          <DropdownMenuItem onClick={handleReuseSettings}>
            <RotateCcw size={14} />
            <span>Reuse Settings</span>
          </DropdownMenuItem>
          <DropdownMenuItem onClick={handleExtend}>
            <FastForward size={14} />
            <span>Extend Video</span>
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>

      <FramePickerDialog
        videoUrl={result.videoUrl}
        open={framePickerOpen}
        onOpenChange={setFramePickerOpen}
        onCapture={handleFrameCapture}
      />
    </>
  );
}
