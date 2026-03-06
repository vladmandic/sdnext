export class ApiError extends Error {
  status: number;
  statusText: string;
  body: unknown;

  constructor(status: number, statusText: string, body: unknown) {
    super(`API Error ${status}: ${statusText}`);
    this.name = "ApiError";
    this.status = status;
    this.statusText = statusText;
    this.body = body;
  }
}

export class ApiClient {
  private baseUrl: string;
  private auth: { username: string; password: string } | null = null;

  constructor(baseUrl?: string) {
    this.baseUrl = baseUrl ?? window.location.origin;
  }

  setBaseUrl(url: string) {
    this.baseUrl = url.replace(/\/$/, "");
  }

  setAuth(username: string, password: string) {
    this.auth = { username, password };
  }

  clearAuth() {
    this.auth = null;
  }

  private getAuthHeaders(): HeadersInit {
    const headers: HeadersInit = {};
    if (this.auth) {
      headers["Authorization"] = `Basic ${btoa(`${this.auth.username}:${this.auth.password}`)}`;
    }
    return headers;
  }

  private getHeaders(): HeadersInit {
    return { "Content-Type": "application/json", ...this.getAuthHeaders() };
  }

  private async request<T>(path: string, options: RequestInit = {}): Promise<T> {
    const url = `${this.baseUrl}${path}`;
    const response = await fetch(url, {
      ...options,
      headers: { ...this.getHeaders(), ...options.headers },
    });
    if (!response.ok) {
      let body: unknown;
      try {
        body = await response.json();
      } catch {
        body = await response.text();
      }
      throw new ApiError(response.status, response.statusText, body);
    }
    const contentType = response.headers.get("content-type");
    if (contentType?.includes("application/json")) {
      return response.json() as Promise<T>;
    }
    return response.text() as unknown as Promise<T>;
  }

  async get<T>(path: string, params?: Record<string, string>, signal?: AbortSignal): Promise<T> {
    const query = params ? `?${new URLSearchParams(params)}` : "";
    return this.request<T>(`${path}${query}`, { method: "GET", signal });
  }

  async post<T>(path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
    return this.request<T>(path, {
      method: "POST",
      body: body != null ? JSON.stringify(body) : undefined,
      signal,
    });
  }

  async delete<T>(path: string, params?: Record<string, string>, signal?: AbortSignal): Promise<T> {
    const query = params ? `?${new URLSearchParams(params)}` : "";
    return this.request<T>(`${path}${query}`, { method: "DELETE", signal });
  }

  async postMultipart<T>(path: string, formData: FormData, signal?: AbortSignal): Promise<T> {
    const url = `${this.baseUrl}${path}`;
    const response = await fetch(url, {
      method: "POST",
      headers: this.getAuthHeaders(),
      body: formData,
      signal,
    });
    if (!response.ok) {
      let body: unknown;
      try {
        body = await response.json();
      } catch {
        body = await response.text();
      }
      throw new ApiError(response.status, response.statusText, body);
    }
    return response.json() as Promise<T>;
  }

  async postBinary(path: string, body?: unknown, signal?: AbortSignal): Promise<Blob> {
    const url = `${this.baseUrl}${path}`;
    const response = await fetch(url, {
      method: "POST",
      headers: this.getHeaders(),
      body: body != null ? JSON.stringify(body) : undefined,
      signal,
    });
    if (!response.ok) {
      throw new ApiError(response.status, response.statusText, null);
    }
    return response.blob();
  }

  getBaseUrl(): string {
    return this.baseUrl;
  }

  getWebSocketUrl(path: string): string {
    const wsProto = this.baseUrl.startsWith("https") ? "wss" : "ws";
    const host = this.baseUrl.replace(/^https?:\/\//, "");
    return `${wsProto}://${host}${path}`;
  }

  async getWsTicket(): Promise<string> {
    const resp = await this.post<{ ticket: string }>("/sdapi/v2/ws-ticket");
    return resp.ticket;
  }
}

export const api = new ApiClient();
