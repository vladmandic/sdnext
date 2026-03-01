export interface Region {
  x: number;
  y: number;
  width: number;
  height: number;
  canvas: HTMLCanvasElement;
}

/** Flood-fill connected component labeling on an RGBA ImageData. */
export function findConnectedComponents(imageData: ImageData, alphaThreshold = 10): Region[] {
  const { width, height, data } = imageData;
  const total = width * height;
  const labels = new Int32Array(total);
  let nextLabel = 1;
  const bounds = new Map<number, { minX: number; minY: number; maxX: number; maxY: number }>();

  for (let i = 0; i < total; i++) {
    if (labels[i] !== 0 || data[i * 4 + 3] < alphaThreshold) continue;

    const label = nextLabel++;
    labels[i] = label;
    const stack = [i];
    let minX = i % width;
    let minY = (i - minX) / width;
    let maxX = minX;
    let maxY = minY;

    while (stack.length > 0) {
      const curr = stack.pop()!;
      const cx = curr % width;
      const cy = (curr - cx) / width;
      if (cx < minX) minX = cx;
      if (cx > maxX) maxX = cx;
      if (cy < minY) minY = cy;
      if (cy > maxY) maxY = cy;

      if (cx > 0 && labels[curr - 1] === 0 && data[(curr - 1) * 4 + 3] >= alphaThreshold) {
        labels[curr - 1] = label;
        stack.push(curr - 1);
      }
      if (cx < width - 1 && labels[curr + 1] === 0 && data[(curr + 1) * 4 + 3] >= alphaThreshold) {
        labels[curr + 1] = label;
        stack.push(curr + 1);
      }
      if (cy > 0 && labels[curr - width] === 0 && data[(curr - width) * 4 + 3] >= alphaThreshold) {
        labels[curr - width] = label;
        stack.push(curr - width);
      }
      if (cy < height - 1 && labels[curr + width] === 0 && data[(curr + width) * 4 + 3] >= alphaThreshold) {
        labels[curr + width] = label;
        stack.push(curr + width);
      }
    }

    bounds.set(label, { minX, minY, maxX, maxY });
  }

  const regions: Region[] = [];
  for (const [label, b] of bounds) {
    const w = b.maxX - b.minX + 1;
    const h = b.maxY - b.minY + 1;
    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d")!;
    const regionData = ctx.createImageData(w, h);

    for (let y = b.minY; y <= b.maxY; y++) {
      for (let x = b.minX; x <= b.maxX; x++) {
        if (labels[y * width + x] !== label) continue;
        const srcAlpha = data[(y * width + x) * 4 + 3];
        const dst = ((y - b.minY) * w + (x - b.minX)) * 4;
        regionData.data[dst] = 255;
        regionData.data[dst + 1] = 255;
        regionData.data[dst + 2] = 255;
        regionData.data[dst + 3] = srcAlpha;
      }
    }

    ctx.putImageData(regionData, 0, 0);
    regions.push({ x: b.minX, y: b.minY, width: w, height: h, canvas });
  }

  return regions;
}
