import { useState, useRef, useCallback } from "react";
import { ChevronLeft, ChevronRight, Camera } from "lucide-react";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";

interface FramePickerDialogProps {
  videoUrl: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onCapture: (blob: Blob, time: number) => void;
}

export function FramePickerDialog({ videoUrl, open, onOpenChange, onCapture }: FramePickerDialogProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [capturing, setCapturing] = useState(false);

  const handleOpenChange = useCallback((next: boolean) => {
    if (!next) {
      setCurrentTime(0);
      setDuration(0);
    }
    onOpenChange(next);
  }, [onOpenChange]);

  const handleLoadedMetadata = useCallback(() => {
    const video = videoRef.current;
    if (video) {
      setDuration(video.duration);
      video.currentTime = 0;
    }
  }, []);

  const handleTimeUpdate = useCallback(() => {
    const video = videoRef.current;
    if (video) setCurrentTime(video.currentTime);
  }, []);

  const handleSliderChange = useCallback((value: number[]) => {
    const video = videoRef.current;
    if (video && value[0] !== undefined) {
      video.currentTime = value[0];
      setCurrentTime(value[0]);
    }
  }, []);

  const stepFrame = useCallback((delta: number) => {
    const video = videoRef.current;
    if (!video) return;
    const frameTime = 1 / 30;
    const newTime = Math.min(Math.max(0, video.currentTime + delta * frameTime), video.duration);
    video.currentTime = newTime;
    setCurrentTime(newTime);
  }, []);

  const handleCapture = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;
    setCapturing(true);
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      setCapturing(false);
      return;
    }
    ctx.drawImage(video, 0, 0);
    canvas.toBlob((blob) => {
      setCapturing(false);
      if (blob) {
        onCapture(blob, video.currentTime);
        handleOpenChange(false);
      }
    }, "image/png");
  }, [onCapture, handleOpenChange]);

  const formatTime = (t: number) => {
    const mins = Math.floor(t / 60);
    const secs = Math.floor(t % 60);
    const ms = Math.floor((t % 1) * 100);
    return `${mins}:${secs.toString().padStart(2, "0")}.${ms.toString().padStart(2, "0")}`;
  };

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle>Extract Frame</DialogTitle>
          <DialogDescription>Scrub to the desired frame and capture it.</DialogDescription>
        </DialogHeader>

        <div className="flex flex-col gap-3">
          <video
            ref={videoRef}
            src={videoUrl}
            className="w-full rounded bg-black"
            onLoadedMetadata={handleLoadedMetadata}
            onTimeUpdate={handleTimeUpdate}
            preload="auto"
            crossOrigin="anonymous"
          />

          <div className="flex items-center gap-2">
            <span className="text-xs tabular-nums text-muted-foreground min-w-[4.5rem]">{formatTime(currentTime)}</span>
            <Slider value={[currentTime]} onValueChange={handleSliderChange} min={0} max={duration || 1} step={0.001} className="flex-1" />
            <span className="text-xs tabular-nums text-muted-foreground min-w-[4.5rem] text-right">{formatTime(duration)}</span>
          </div>

          <div className="flex items-center justify-between">
            <div className="flex items-center gap-1">
              <Button variant="outline" size="icon-sm" onClick={() => stepFrame(-1)} title="Previous frame">
                <ChevronLeft size={14} />
              </Button>
              <Button variant="outline" size="icon-sm" onClick={() => stepFrame(1)} title="Next frame">
                <ChevronRight size={14} />
              </Button>
            </div>

            <Button onClick={handleCapture} disabled={capturing || duration === 0} size="sm">
              <Camera size={14} />
              Capture Frame
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
