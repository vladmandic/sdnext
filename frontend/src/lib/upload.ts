import { api } from "@/api/client";
import type { UploadResponse } from "@/api/types/upload";

export async function uploadFile(file: File): Promise<string> {
  const form = new FormData();
  form.append("files", file);
  const res = await api.postMultipart<UploadResponse>("/sdapi/v2/upload", form);
  return res.uploads[0].ref;
}

export async function uploadFiles(files: File[]): Promise<string[]> {
  if (files.length === 0) return [];
  const form = new FormData();
  for (const file of files) {
    form.append("files", file);
  }
  const res = await api.postMultipart<UploadResponse>("/sdapi/v2/upload", form);
  return res.uploads.map((u) => u.ref);
}

export async function uploadBlob(blob: Blob, filename = "upload.png"): Promise<string> {
  const form = new FormData();
  form.append("files", blob, filename);
  const res = await api.postMultipart<UploadResponse>("/sdapi/v2/upload", form);
  return res.uploads[0].ref;
}
