type EventHandler = (data: unknown) => void;
type BinaryHandler = (data: ArrayBuffer) => void;

interface WsEvents {
  message: EventHandler;
  binary: BinaryHandler;
  open: () => void;
  close: (event: CloseEvent) => void;
  error: (event: Event) => void;
  max_retries: () => void;
}

export class WebSocketManager {
  private ws: WebSocket | null = null;
  private url: string;
  private ticketFn: (() => Promise<string>) | null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 10;
  private reconnectDelay = 1000;
  private maxReconnectDelay = 30_000;
  private shouldReconnect = true;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private listeners = new Map<keyof WsEvents, Set<(...args: never[]) => void>>();

  constructor(url: string, ticketFn?: () => Promise<string>) {
    this.url = url;
    this.ticketFn = ticketFn ?? null;
  }

  connect(): void {
    if (this.ws?.readyState === WebSocket.OPEN || this.ws?.readyState === WebSocket.CONNECTING) return;

    if (this.ticketFn) {
      this.ticketFn()
        .then((ticket) => this.openSocket(`${this.url}${this.url.includes("?") ? "&" : "?"}ticket=${ticket}`))
        .catch(() => this.openSocket(this.url));
    } else {
      this.openSocket(this.url);
    }
  }

  private openSocket(url: string): void {
    if (this.ws?.readyState === WebSocket.OPEN || this.ws?.readyState === WebSocket.CONNECTING) return;

    this.ws = new WebSocket(url);
    this.ws.binaryType = "arraybuffer";

    this.ws.onopen = () => {
      this.reconnectAttempts = 0;
      this.shouldReconnect = true;
      this.emit("open");
    };

    this.ws.onmessage = (event: MessageEvent) => {
      if (event.data instanceof ArrayBuffer) {
        this.emit("binary", event.data);
      } else {
        try {
          const data = JSON.parse(event.data);
          this.emit("message", data);
        } catch {
          this.emit("message", event.data);
        }
      }
    };

    this.ws.onclose = (event: CloseEvent) => {
      this.emit("close", event);
      // Don't retry on explicit policy-violation close (auth/forbidden)
      if (event.code === 1008) {
        this.shouldReconnect = false;
      }
      // Don't retry on custom application close codes (e.g. 4004 "Job not found")
      if (event.code >= 4000 && event.code < 5000) {
        this.shouldReconnect = false;
      }
      if (this.shouldReconnect) {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
          this.shouldReconnect = false;
          this.emit("max_retries");
          return;
        }
        const delay = Math.min(this.reconnectDelay * Math.pow(2, this.reconnectAttempts), this.maxReconnectDelay);
        this.reconnectAttempts++;
        this.reconnectTimer = setTimeout(() => {
          this.reconnectTimer = null;
          if (this.shouldReconnect) this.connect();
        }, delay);
      }
    };

    this.ws.onerror = (event: Event) => {
      this.emit("error", event);
    };
  }

  disconnect(): void {
    this.shouldReconnect = false;
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.ws?.close();
    this.ws = null;
  }

  updateUrl(url: string): void {
    this.disconnect();
    this.url = url;
    this.reconnectAttempts = 0;
    this.shouldReconnect = true;
    this.connect();
  }

  send(data: string | Record<string, unknown>): void {
    if (this.ws?.readyState !== WebSocket.OPEN) return;
    const payload = typeof data === "string" ? data : JSON.stringify(data);
    this.ws.send(payload);
  }

  sendBinary(data: ArrayBuffer): void {
    if (this.ws?.readyState !== WebSocket.OPEN) return;
    this.ws.send(data);
  }

  on<K extends keyof WsEvents>(event: K, handler: WsEvents[K]): () => void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)!.add(handler);
    return () => this.listeners.get(event)?.delete(handler);
  }

  get connected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  private emit(event: keyof WsEvents, ...args: unknown[]): void {
    this.listeners.get(event)?.forEach((handler) => {
      (handler as (...a: unknown[]) => void)(...args);
    });
  }
}
