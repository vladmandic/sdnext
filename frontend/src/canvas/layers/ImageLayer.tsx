import { useEffect, useState } from "react";
import { Layer, Image as KonvaImage } from "react-konva";
import { useImg2ImgStore } from "@/stores/img2imgStore";

export function ImageLayer() {
  const initImageData = useImg2ImgStore((s) => s.initImageData);
  const [image, setImage] = useState<HTMLImageElement | null>(null);

  useEffect(() => {
    if (!initImageData) {
      setImage(null);
      return;
    }
    const img = new window.Image();
    img.onload = () => setImage(img);
    img.src = initImageData;
  }, [initImageData]);

  if (!image) return null;

  return (
    <Layer>
      <KonvaImage image={image} x={0} y={0} />
    </Layer>
  );
}
