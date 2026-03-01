import { useCallback, useRef, useState, type MouseEvent as ReactMouseEvent } from "react";
import { Play, Pause, Maximize, Minimize, SkipBack, SkipForward } from "lucide-react";
import { cn } from "@/lib/utils";

const SPEEDS = [0.25, 0.5, 1, 2, 4] as const;

interface PlayerControlsProps {
  playing: boolean;
  currentTime: number;
  duration: number;
  speed: number;
  isFullscreen: boolean;
  fps: number;
  width: number;
  height: number;
  visible: boolean;
  trackWidth: number;
  onTogglePlay: () => void;
  onSeek: (time: number) => void;
  onStepFrame: (delta: number) => void;
  onSetSpeed: (speed: number) => void;
  onToggleFullscreen: () => void;
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export function PlayerControls({
  playing,
  currentTime,
  duration,
  speed,
  isFullscreen,
  fps,
  width,
  height,
  visible,
  trackWidth,
  onTogglePlay,
  onSeek,
  onStepFrame,
  onSetSpeed,
  onToggleFullscreen,
}: PlayerControlsProps) {
  const trackRef = useRef<HTMLDivElement>(null);
  const [dragging, setDragging] = useState(false);
  const [hoverX, setHoverX] = useState<number | null>(null);

  const currentFrame = Math.floor(currentTime * fps);
  const totalFrames = Math.max(1, Math.floor(duration * fps));
  const progressFraction = duration > 0 ? currentTime / duration : 0;

  const seekFromMouse = useCallback(
    (clientX: number) => {
      const track = trackRef.current;
      if (!track || duration <= 0) return;
      const rect = track.getBoundingClientRect();
      const fraction = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
      onSeek(fraction * duration);
    },
    [duration, onSeek],
  );

  const handleTrackMouseDown = useCallback(
    (e: ReactMouseEvent) => {
      e.preventDefault();
      setDragging(true);
      seekFromMouse(e.clientX);

      const onMove = (ev: globalThis.MouseEvent) => seekFromMouse(ev.clientX);
      const onUp = () => {
        setDragging(false);
        window.removeEventListener("mousemove", onMove);
        window.removeEventListener("mouseup", onUp);
      };
      window.addEventListener("mousemove", onMove);
      window.addEventListener("mouseup", onUp);
    },
    [seekFromMouse],
  );

  const handleTrackHover = useCallback(
    (e: ReactMouseEvent) => {
      const track = trackRef.current;
      if (!track) return;
      const rect = track.getBoundingClientRect();
      setHoverX(e.clientX - rect.left);
    },
    [],
  );

  const cycleSpeed = useCallback(() => {
    const idx = SPEEDS.indexOf(speed as (typeof SPEEDS)[number]);
    const next = SPEEDS[(idx + 1) % SPEEDS.length];
    onSetSpeed(next);
  }, [speed, onSetSpeed]);

  const hoverTimestamp = hoverX !== null && trackWidth > 0
    ? formatTime((hoverX / trackWidth) * duration)
    : null;

  return (
    <div
      className={cn(
        "absolute inset-x-0 bottom-0 transition-opacity duration-200 pointer-events-none",
        visible || dragging ? "opacity-100" : "opacity-0",
      )}
      onClick={(e) => e.stopPropagation()}
    >
      <div className="pointer-events-auto mx-3 mb-3 rounded-lg bg-black/70 backdrop-blur-sm px-3 py-2 flex flex-col gap-1.5">
        {/* Scrub bar */}
        <div
          ref={trackRef}
          className="relative h-5 flex items-center cursor-pointer group"
          onMouseDown={handleTrackMouseDown}
          onMouseMove={handleTrackHover}
          onMouseLeave={() => setHoverX(null)}
        >
          {/* Track background */}
          <div className="absolute inset-x-0 h-1 bg-white/20 rounded-full group-hover:h-1.5 transition-all">
            {/* Progress fill */}
            <div
              className="absolute inset-y-0 left-0 bg-primary rounded-full"
              style={{ width: `${progressFraction * 100}%` }}
            />
            {/* Drag handle */}
            <div
              className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-3 h-3 bg-primary rounded-full opacity-0 group-hover:opacity-100 transition-opacity shadow"
              style={{ left: `${progressFraction * 100}%` }}
            />
          </div>
          {/* Hover timestamp tooltip */}
          {hoverX !== null && hoverTimestamp && (
            <div
              className="absolute -top-7 -translate-x-1/2 px-1.5 py-0.5 bg-black/90 text-white text-3xs rounded pointer-events-none"
              style={{ left: hoverX }}
            >
              {hoverTimestamp}
            </div>
          )}
        </div>

        {/* Controls row */}
        <div className="flex items-center gap-2 text-white text-xs">
          {/* Frame step back */}
          <button onClick={() => onStepFrame(-1)} className="hover:text-primary transition-colors" title="Previous frame (Left)">
            <SkipBack size={14} />
          </button>

          {/* Play/Pause */}
          <button onClick={onTogglePlay} className="hover:text-primary transition-colors" title="Play/Pause (Space)">
            {playing ? <Pause size={16} /> : <Play size={16} />}
          </button>

          {/* Frame step forward */}
          <button onClick={() => onStepFrame(1)} className="hover:text-primary transition-colors" title="Next frame (Right)">
            <SkipForward size={14} />
          </button>

          {/* Time display */}
          <span className="tabular-nums text-white/80 text-3xs select-none">
            {formatTime(currentTime)} / {formatTime(duration)}
          </span>

          <div className="flex-1" />

          {/* Frame counter */}
          <span className="tabular-nums text-white/60 text-3xs select-none hidden sm:inline">
            {currentFrame} / {totalFrames}f
          </span>

          {/* Resolution */}
          {width > 0 && height > 0 && (
            <span className="text-white/60 text-3xs select-none hidden sm:inline">
              {width}x{height}
            </span>
          )}

          {/* Speed */}
          <button
            onClick={cycleSpeed}
            className="tabular-nums text-3xs px-1.5 py-0.5 rounded bg-white/10 hover:bg-white/20 transition-colors select-none"
            title="Playback speed ([/])"
          >
            {speed}x
          </button>

          {/* Fullscreen */}
          <button onClick={onToggleFullscreen} className="hover:text-primary transition-colors" title="Fullscreen (F)">
            {isFullscreen ? <Minimize size={14} /> : <Maximize size={14} />}
          </button>
        </div>
      </div>
    </div>
  );
}
