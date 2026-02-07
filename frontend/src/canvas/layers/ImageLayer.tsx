import { useEffect, useState } from "react";
import { Layer, Image as KonvaImage, Rect, Label, Tag, Text } from "react-konva";
import { useImg2ImgStore } from "@/stores/img2imgStore";
import { contrastText } from "@/lib/utils";

const BORDER_COLOR = "#4ade80";
const LABEL_HEIGHT = 19; // fontSize(11) + padding(4)*2

export function ImageLayer() {
  const initImageData = useImg2ImgStore((s) => s.initImageData);
  const initW = useImg2ImgStore((s) => s.initImageWidth);
  const initH = useImg2ImgStore((s) => s.initImageHeight);
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
    <Layer listening={false}>
      <KonvaImage image={image} x={0} y={0} />
      <Rect
        x={0}
        y={0}
        width={initW}
        height={initH}
        stroke={BORDER_COLOR}
        strokeWidth={2}
        listening={false}
      />
      <Label x={0} y={-LABEL_HEIGHT} listening={false}>
        <Tag fill={BORDER_COLOR} cornerRadius={3} />
        <Text text="Input" fontSize={11} fill={contrastText(BORDER_COLOR)} padding={4} listening={false} />
      </Label>
    </Layer>
  );
}
