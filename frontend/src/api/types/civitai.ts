export interface CivitImage {
  id: number;
  url: string;
  width: number;
  height: number;
  type: string;
  nsfwLevel: number;
}

export interface CivitFileHashes {
  SHA256: string | null;
  AutoV1: string | null;
  AutoV2: string | null;
  AutoV3: string | null;
  CRC32: string | null;
  BLAKE3: string | null;
}

export interface CivitFile {
  id: number;
  name: string;
  type: string;
  sizeKB: number;
  hashes: CivitFileHashes;
  downloadUrl: string;
  primary: boolean | null;
}

export interface CivitStats {
  downloadCount: number;
  favoriteCount: number;
  thumbsUpCount: number;
  thumbsDownCount: number;
  commentCount: number;
  ratingCount: number;
  rating: number;
}

export interface CivitVersion {
  id: number;
  modelId: number;
  name: string;
  baseModel: string;
  publishedAt: string | null;
  availability: string;
  description: string | null;
  trainedWords: string[];
  stats: CivitStats;
  files: CivitFile[];
  images: CivitImage[];
  nsfwLevel: number;
  downloadUrl: string;
}

export interface CivitCreator {
  username: string;
  image: string | null;
}

export interface CivitModel {
  id: number;
  type: string;
  name: string;
  description: string | null;
  tags: string[];
  nsfw: boolean;
  nsfwLevel: number;
  availability: string;
  stats: CivitStats;
  creator: CivitCreator;
  modelVersions: CivitVersion[];
  allowNoCredit: boolean;
  allowCommercialUse: string[];
  allowDerivatives: boolean;
  allowDifferentLicense: boolean;
}

export interface CivitSearchMetadata {
  nextPage: string | null;
  currentPage: number | null;
  pageSize: number | null;
  totalPages: number | null;
  totalItems: number | null;
  nextCursor: string | null;
}

export interface CivitSearchResponse {
  items: CivitModel[];
  metadata: CivitSearchMetadata;
  requestUrl: string | null;
}

export interface CivitOptions {
  types: string[];
  sort: string[];
  period: string[];
  base_models: string[];
}

export interface CivitDownloadItem {
  id: string;
  url: string;
  filename: string;
  folder: string;
  model_type: string;
  status: string;
  progress: number;
  bytes_downloaded: number;
  bytes_total: number;
  error: string | null;
  created_at: string;
  completed_at: string | null;
}

export interface CivitDownloadStatus {
  active: CivitDownloadItem[];
  queued: CivitDownloadItem[];
  completed: CivitDownloadItem[];
}

export interface CivitDownloadRequest {
  url: string;
  filename?: string;
  folder?: string;
  model_type?: string;
  expected_hash?: string;
  token?: string;
  model_name?: string;
  base_model?: string;
  creator?: string;
  model_id?: number;
  version_id?: number;
  version_name?: string;
  nsfw?: boolean;
}

export interface CivitSettings {
  token_configured: boolean;
  save_subfolder: string;
  save_type_folders: string;
  discard_hash_mismatch: boolean;
  download_workers: number;
}

export interface CivitSettingsUpdate {
  token?: string;
  save_subfolder?: string;
  discard_hash_mismatch?: boolean;
}

export interface CivitHistoryEntry {
  type: string;
  term: string;
  timestamp: string;
}

export interface CivitSearchParams {
  query?: string;
  tag?: string;
  types?: string;
  sort?: string;
  period?: string;
  base_models?: string;
  nsfw?: boolean;
  limit?: number;
  cursor?: string;
}
