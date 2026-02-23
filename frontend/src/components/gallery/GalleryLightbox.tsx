import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useGalleryStore } from "@/stores/galleryStore";
import { X, ChevronLeft, ChevronRight, ZoomIn, ZoomOut, RotateCcw } from "lucide-react";

export function GalleryLightbox() {
  const lightboxIndex = useGalleryStore((s) => s.lightboxIndex);
  const files = useGalleryStore((s) => s.sortedFiles);
  const thumbs = useGalleryStore((s) => s.thumbs);
  const closeLightbox = useGalleryStore((s) => s.closeLightbox);
  const navigateLightbox = useGalleryStore((s) => s.navigateLightbox);
  const selectFile = useGalleryStore((s) => s.selectFile);

  const [scale, setScale] = useState(1);
  const [translate, setTranslate] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [prevIndex, setPrevIndex] = useState(lightboxIndex);
  const dragStart = useRef({ x: 0, y: 0 });
  const imgRef = useRef<HTMLDivElement>(null);

  const isOpen = lightboxIndex !== null;
  const file = isOpen ? files[lightboxIndex] : null;
  const thumb = file ? thumbs.get(file.id) : null;
  const maxIndex = files.length - 1;

  // Full-size image URL
  const fullUrl = useMemo(() => {
    if (!file) return null;
    return `/file=${file.fullPath}`;
  }, [file]);

  // Reset transform on navigation (adjust state during render pattern)
  if (prevIndex !== lightboxIndex) {
    setPrevIndex(lightboxIndex);
    setScale(1);
    setTranslate({ x: 0, y: 0 });
  }

  // Sync selection
  useEffect(() => {
    if (file && thumb) selectFile(file, thumb);
  }, [file, thumb, selectFile]);

  const navigate = useCallback((delta: number) => {
    navigateLightbox(delta, maxIndex);
  }, [navigateLightbox, maxIndex]);

  const resetTransform = useCallback(() => {
    setScale(1);
    setTranslate({ x: 0, y: 0 });
  }, []);

  // Keyboard
  useEffect(() => {
    if (!isOpen) return;
    const handler = (e: KeyboardEvent) => {
      switch (e.key) {
        case "Escape": closeLightbox(); break;
        case "ArrowLeft": navigate(-1); break;
        case "ArrowRight": navigate(1); break;
        case "+": case "=": setScale((s) => Math.min(8, s * 1.25)); break;
        case "-": setScale((s) => Math.max(0.25, s / 1.25)); break;
        case "0": resetTransform(); break;
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [isOpen, closeLightbox, navigate, resetTransform]);

  // Mouse wheel zoom
  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
    setScale((s) => Math.max(0.25, Math.min(8, s * factor)));
  }, []);

  // Pan handlers
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (scale <= 1) return;
    setIsDragging(true);
    dragStart.current = { x: e.clientX - translate.x, y: e.clientY - translate.y };
  }, [scale, translate]);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!isDragging) return;
    setTranslate({ x: e.clientX - dragStart.current.x, y: e.clientY - dragStart.current.y });
  }, [isDragging]);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  if (!isOpen || !file) return null;

  const filename = file.relativePath.split("/").pop() ?? file.relativePath;

  return (
    <div className="fixed inset-0 z-50 bg-black/90 flex flex-col" onClick={closeLightbox}>
      {/* Top bar */}
      <div className="flex items-center justify-between px-4 py-2 flex-shrink-0" onClick={(e) => e.stopPropagation()}>
        <span className="text-xs text-white/70 truncate max-w-[50%]">{filename}</span>
        <div className="flex items-center gap-1">
          <LightboxButton onClick={() => setScale((s) => Math.min(8, s * 1.25))}><ZoomIn size={16} /></LightboxButton>
          <LightboxButton onClick={() => setScale((s) => Math.max(0.25, s / 1.25))}><ZoomOut size={16} /></LightboxButton>
          <LightboxButton onClick={resetTransform}><RotateCcw size={16} /></LightboxButton>
          <span className="text-3xs text-white/50 tabular-nums w-10 text-center">{Math.round(scale * 100)}%</span>
          <LightboxButton onClick={closeLightbox}><X size={16} /></LightboxButton>
        </div>
      </div>

      {/* Image area */}
      <div
        ref={imgRef}
        className="flex-1 flex items-center justify-center overflow-hidden select-none"
        style={{ cursor: scale > 1 ? (isDragging ? "grabbing" : "grab") : "default" }}
        onWheel={handleWheel}
        onMouseDown={(e) => { e.stopPropagation(); handleMouseDown(e); }}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onClick={(e) => e.stopPropagation()}
      >
        {fullUrl && (
          <img
            src={fullUrl}
            alt={filename}
            className="max-w-full max-h-full object-contain transition-transform duration-100"
            style={{ transform: `translate(${translate.x}px, ${translate.y}px) scale(${scale})` }}
            draggable={false}
          />
        )}
      </div>

      {/* Navigation arrows */}
      {lightboxIndex > 0 && (
        <button
          className="absolute left-2 top-1/2 -translate-y-1/2 w-10 h-10 flex items-center justify-center rounded-full bg-black/50 text-white/70 hover:text-white hover:bg-black/70 transition-colors"
          onClick={(e) => { e.stopPropagation(); navigate(-1); }}
        >
          <ChevronLeft size={24} />
        </button>
      )}
      {lightboxIndex < maxIndex && (
        <button
          className="absolute right-2 top-1/2 -translate-y-1/2 w-10 h-10 flex items-center justify-center rounded-full bg-black/50 text-white/70 hover:text-white hover:bg-black/70 transition-colors"
          onClick={(e) => { e.stopPropagation(); navigate(1); }}
        >
          <ChevronRight size={24} />
        </button>
      )}

      {/* Bottom bar */}
      <div className="flex items-center justify-center px-4 py-1.5 flex-shrink-0" onClick={(e) => e.stopPropagation()}>
        <span className="text-3xs text-white/40 tabular-nums">
          {lightboxIndex + 1} / {files.length}
          {thumb && ` | ${thumb.width}x${thumb.height}`}
        </span>
      </div>
    </div>
  );
}

function LightboxButton({ children, onClick }: { children: React.ReactNode; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="w-8 h-8 flex items-center justify-center rounded text-white/70 hover:text-white hover:bg-white/10 transition-colors"
    >
      {children}
    </button>
  );
}
