export interface VideoResult {
  id: string;
  videoUrl: string;
  thumbnailUrl?: string;
  width: number;
  height: number;
  format: string;
  size: number;
  params: Record<string, unknown>;
  domain: "video" | "framepack" | "ltx";
  timestamp: number;
}

export type VideoMode = "t2v" | "i2v" | "flf2v" | "vace" | "animate";

export interface VideoModelDetail {
  name: string;
  repo: string;
  url: string;
  cached: boolean;
  loaded: boolean;
  mode: VideoMode;
}

export interface VideoEngineModel {
  name: string;
  repo: string;
  url: string;
}

export interface VideoEngine {
  engine: string;
  models: string[];
  model_details: VideoModelDetail[];
}

export interface VideoLoadResponse {
  engine: string;
  model: string;
  messages: string[];
}
