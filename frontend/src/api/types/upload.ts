export interface UploadRef {
  ref: string;
  url: string;
  name: string;
  size: number;
}

export interface UploadResponse {
  uploads: UploadRef[];
}
