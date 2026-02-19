import { Film } from "lucide-react";

interface VideoPlayerProps {
  src: string | null;
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

  return (
    <div className="flex items-center justify-center h-full p-4">
      <video
        src={src}
        controls
        loop
        className="max-w-full max-h-full rounded"
      />
    </div>
  );
}
