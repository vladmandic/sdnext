export interface BrowserFolder {
  path: string;
  label: string;
}

export interface BrowserSubdir {
  path: string;
  label: string;
}

export interface BrowserThumb {
  exif: string;
  data: string;
  width: number;
  height: number;
  size: number;
  mtime: number;
}

export interface GalleryFile {
  folder: string;
  relativePath: string;
  fullPath: string;
  id: string;
}

export interface ParsedGenerationInfo {
  prompt: string;
  negativePrompt: string;
  params: Record<string, string>;
}

export type GallerySortField = "name" | "mtime" | "size" | "width";
export type GallerySortDir = "asc" | "desc";
export interface GallerySort {
  field: GallerySortField;
  dir: GallerySortDir;
}

export interface CachedThumb {
  hash: string;
  folder: string;
  data: string;
  width: number;
  height: number;
  size: number;
  mtime: number;
  exif: string;
}
