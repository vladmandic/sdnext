import { useCallback, useEffect, useRef, useState } from "react";
import { Film } from "lucide-react";
import { PlayerControls } from "@/components/video/PlayerControls";

const SPEEDS = [0.25, 0.5, 1, 2, 4] as const;
const IDLE_TIMEOUT_MS = 2500;
const DEFAULT_FPS = 24;

interface VideoPlayerProps {
  src: string | null;
}

function VideoPlayerInner({ src }: { src: string }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const idleTimerRef = useRef<ReturnType<typeof setTimeout>>(undefined);

  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [speed, setSpeed] = useState<number>(1);
  const [showControls, setShowControls] = useState(true);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [meta, setMeta] = useState({ fps: DEFAULT_FPS, width: 0, height: 0 });
  const [trackWidth, setTrackWidth] = useState(0);

  const startIdleTimer = useCallback(() => {
    clearTimeout(idleTimerRef.current);
    idleTimerRef.current = setTimeout(() => setShowControls(false), IDLE_TIMEOUT_MS);
  }, []);

  const resetIdleTimer = useCallback(() => {
    setShowControls(true);
    startIdleTimer();
  }, [startIdleTimer]);

  // Video event handlers
  const handleLoadedMetadata = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;
    setDuration(video.duration);
    setMeta({ fps: DEFAULT_FPS, width: video.videoWidth, height: video.videoHeight });
  }, []);

  const handleTimeUpdate = useCallback(() => {
    if (videoRef.current) setCurrentTime(videoRef.current.currentTime);
  }, []);

  const handlePlay = useCallback(() => {
    setPlaying(true);
    startIdleTimer();
  }, [startIdleTimer]);

  const handlePause = useCallback(() => {
    setPlaying(false);
    clearTimeout(idleTimerRef.current);
    setShowControls(true);
  }, []);

  const handleEnded = useCallback(() => {
    setPlaying(false);
    clearTimeout(idleTimerRef.current);
    setShowControls(true);
  }, []);

  const togglePlay = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused) video.play();
    else video.pause();
  }, []);

  const seek = useCallback((time: number) => {
    if (videoRef.current) {
      videoRef.current.currentTime = Math.max(0, Math.min(time, videoRef.current.duration || 0));
    }
  }, []);

  const stepFrame = useCallback(
    (delta: number) => {
      const video = videoRef.current;
      if (!video) return;
      video.pause();
      const frameDuration = 1 / meta.fps;
      video.currentTime = Math.max(0, Math.min(video.duration, video.currentTime + delta * frameDuration));
    },
    [meta.fps],
  );

  const setPlaybackSpeed = useCallback((s: number) => {
    setSpeed(s);
    if (videoRef.current) videoRef.current.playbackRate = s;
  }, []);

  const toggleFullscreen = useCallback(() => {
    const container = containerRef.current;
    if (!container) return;
    if (document.fullscreenElement) {
      document.exitFullscreen();
    } else {
      container.requestFullscreen();
    }
  }, []);

  // Track container width for scrub bar hover timestamp
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) setTrackWidth(entry.contentRect.width - 24);
    });
    ro.observe(container);
    return () => ro.disconnect();
  }, []);

  // Fullscreen change listener
  useEffect(() => {
    const handleFsChange = () => setIsFullscreen(!!document.fullscreenElement);
    document.addEventListener("fullscreenchange", handleFsChange);
    return () => document.removeEventListener("fullscreenchange", handleFsChange);
  }, []);

  // Keyboard shortcuts
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;

      const video = videoRef.current;
      if (!video) return;

      switch (e.key) {
        case " ":
          e.preventDefault();
          togglePlay();
          break;
        case "ArrowLeft":
          e.preventDefault();
          stepFrame(-1);
          break;
        case "ArrowRight":
          e.preventDefault();
          stepFrame(1);
          break;
        case "j":
        case "J":
          video.currentTime = Math.max(0, video.currentTime - 2);
          break;
        case "k":
        case "K":
          video.pause();
          break;
        case "l":
        case "L":
          video.currentTime = Math.min(video.duration, video.currentTime + 2);
          break;
        case "f":
        case "F":
          toggleFullscreen();
          break;
        case "[": {
          const idx = SPEEDS.indexOf(speed as (typeof SPEEDS)[number]);
          if (idx > 0) setPlaybackSpeed(SPEEDS[idx - 1]);
          break;
        }
        case "]": {
          const idx = SPEEDS.indexOf(speed as (typeof SPEEDS)[number]);
          if (idx < SPEEDS.length - 1) setPlaybackSpeed(SPEEDS[idx + 1]);
          break;
        }
        default:
          if (e.key >= "0" && e.key <= "9") {
            const fraction = parseInt(e.key) / 10;
            video.currentTime = fraction * video.duration;
          }
          break;
      }
    };

    container.addEventListener("keydown", handleKeyDown);
    return () => container.removeEventListener("keydown", handleKeyDown);
  }, [speed, togglePlay, stepFrame, setPlaybackSpeed, toggleFullscreen]);

  return (
    <div
      ref={containerRef}
      className="relative h-full w-full outline-none"
      tabIndex={0}
      onMouseMove={resetIdleTimer}
      onClick={togglePlay}
    >
      <video
        ref={videoRef}
        src={src}
        loop
        className="w-full h-full object-contain"
        onLoadedMetadata={handleLoadedMetadata}
        onTimeUpdate={handleTimeUpdate}
        onPlay={handlePlay}
        onPause={handlePause}
        onEnded={handleEnded}
      />
      <PlayerControls
        playing={playing}
        currentTime={currentTime}
        duration={duration}
        speed={speed}
        isFullscreen={isFullscreen}
        fps={meta.fps}
        width={meta.width}
        height={meta.height}
        visible={showControls}
        trackWidth={trackWidth}
        onTogglePlay={togglePlay}
        onSeek={seek}
        onStepFrame={stepFrame}
        onSetSpeed={setPlaybackSpeed}
        onToggleFullscreen={toggleFullscreen}
      />
    </div>
  );
}

export function VideoPlayer({ src }: VideoPlayerProps) {
  if (!src) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-muted-foreground">
        <Film size={48} className="mb-3 opacity-30" />
        <p className="text-sm opacity-50">Video result will appear here</p>
      </div>
    );
  }

  // key={src} resets all state when the source changes, avoiding effects that set state
  return <VideoPlayerInner key={src} src={src} />;
}
