export interface VideoEngineModel {
  name: string;
  repo: string;
  url: string;
}

export interface VideoEngine {
  engine: string;
  models: string[];
}

export interface VideoLoadResponse {
  engine: string;
  model: string;
  messages: string[];
}
