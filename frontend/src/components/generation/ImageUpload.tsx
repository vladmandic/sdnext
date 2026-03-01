import { useCallback, useEffect, useRef, useState } from "react";
import { Upload, X } from "lucide-react";
import { INTERNAL_MIME } from "@/stores/dragStore";
import { payloadToFile } from "@/lib/sendTo";
import type { DragPayload } from "@/stores/dragStore";
import { Button } from "@/components/ui/button";

interface ImageUploadProps {
  image: File | null;
  onImageChange: (file: File | null) => void;
  label?: string;
  compact?: boolean;
}

export function ImageUpload({ image, onImageChange, label = "Drop image", compact = false }: ImageUploadProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const [preview, setPreview] = useState<string | null>(() => (image ? URL.createObjectURL(image) : null));

  // Revoke object URL on unmount
  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview);
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps -- cleanup only on unmount

  const handleFile = useCallback((file: File | null) => {
    if (preview) URL.revokeObjectURL(preview);
    if (file) {
      setPreview(URL.createObjectURL(file));
    } else {
      setPreview(null);
    }
    onImageChange(file);
  }, [onImageChange, preview]);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    // Check for internal drag payload first
    const raw = e.dataTransfer.getData(INTERNAL_MIME);
    if (raw) {
      try {
        const payload = JSON.parse(raw) as DragPayload;
        payloadToFile(payload).then((f) => handleFile(f)).catch(() => {});
        return;
      } catch { /* fall through */ }
    }
    const file = e.dataTransfer.files?.[0];
    if (file?.type.startsWith("image/")) handleFile(file);
  }, [handleFile]);

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(true);
  }, []);

  const onDragLeave = useCallback(() => setDragOver(false), []);

  const onInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] ?? null;
    handleFile(file);
    e.target.value = "";
  }, [handleFile]);

  const size = compact ? "h-20" : "h-28";

  if (image && preview) {
    return (
      <div className={`relative ${size} rounded-md overflow-hidden border border-border group`}>
        <img src={preview} alt="Upload preview" className="w-full h-full object-cover" />
        <Button
          variant="destructive"
          size="icon-sm"
          className="absolute top-1 right-1 opacity-0 group-hover:opacity-100 transition-opacity h-5 w-5"
          onClick={() => handleFile(null)}
        >
          <X size={10} />
        </Button>
      </div>
    );
  }

  return (
    <>
      <div
        className={`${size} rounded-md border-2 border-dashed flex flex-col items-center justify-center cursor-pointer text-muted-foreground hover:text-foreground hover:border-foreground/30 transition-colors ${dragOver ? "border-primary bg-primary/5" : "border-border"}`}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onClick={() => inputRef.current?.click()}
      >
        <Upload size={compact ? 14 : 16} className="mb-1" />
        <span className="text-3xs">{label}</span>
      </div>
      <input ref={inputRef} type="file" accept="image/*" className="hidden" onChange={onInputChange} />
    </>
  );
}
