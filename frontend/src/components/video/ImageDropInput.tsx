import { useCallback, useState, useRef, useMemo } from "react";
import { X, ImagePlus } from "lucide-react";

interface ImageDropInputProps {
  label: string;
  value: File | null;
  onChange: (value: File | null) => void;
}

export function ImageDropInput({ label, value, onChange }: ImageDropInputProps) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback((file: File) => {
    if (!file.type.startsWith("image/")) return;
    onChange(file);
  }, [onChange]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }, [handleFile]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(true);
  }, []);

  const handleDragLeave = useCallback(() => setDragging(false), []);

  const handleClick = useCallback(() => inputRef.current?.click(), []);

  const handleInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
    e.target.value = "";
  }, [handleFile]);

  const handleRemove = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    onChange(null);
  }, [onChange]);

  const previewUrl = useMemo(() => (value ? URL.createObjectURL(value) : null), [value]);

  if (value && previewUrl) {
    return (
      <div className="space-y-1">
        <span className="text-2xs text-muted-foreground">{label}</span>
        <div className="relative group w-full h-20 rounded border border-border overflow-hidden cursor-pointer" onClick={handleClick}>
          <img src={previewUrl} alt={label} className="w-full h-full object-cover" />
          <button
            type="button"
            onClick={handleRemove}
            className="absolute top-1 right-1 p-0.5 rounded bg-background/80 text-foreground opacity-0 group-hover:opacity-100 transition-opacity"
          >
            <X size={12} />
          </button>
          <input ref={inputRef} type="file" accept="image/*" className="hidden" onChange={handleInputChange} />
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-1">
      <span className="text-2xs text-muted-foreground">{label}</span>
      <div
        className={`flex flex-col items-center justify-center w-full h-20 rounded border-2 border-dashed cursor-pointer transition-colors ${dragging ? "border-primary bg-primary/5" : "border-muted-foreground/25 hover:border-muted-foreground/50"}`}
        onClick={handleClick}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
      >
        <ImagePlus size={16} className="text-muted-foreground/50 mb-1" />
        <span className="text-3xs text-muted-foreground/50">Drop or click</span>
        <input ref={inputRef} type="file" accept="image/*" className="hidden" onChange={handleInputChange} />
      </div>
    </div>
  );
}
