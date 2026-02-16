type EventHandler = (data: unknown) => void;
type BinaryHandler = (data: ArrayBuffer) => void;

interface WsEvents {
  message: EventHandler;
  binary: BinaryHandler;
  open: () => void;
  close: (event: CloseEvent) => void;
  error: (event: Event) => void;
}

export class WebSocketManager {
  private ws: WebSocket | null = null;
  private url: string;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 10;
  private reconnectDelay = 1000;
  private shouldReconnect = true;
  private listeners = new Map<keyof WsEvents, Set<(...args: never[]) => void>>();

  constructor(url: string) {
    this.url = url;
  }

  connect(): void {
    if (this.ws?.readyState === WebSocket.OPEN) return;

    this.ws = new WebSocket(this.url);
    this.ws.binaryType = "arraybuffer";

    this.ws.onopen = () => {
      this.reconnectAttempts = 0;
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
      // Don't retry on auth/forbidden errors (code 1008 or HTTP 403 mapped to 1006)
      if (event.code === 1008 || (event.code === 1006 && this.reconnectAttempts > 0)) {
        this.shouldReconnect = false;
      }
      if (this.shouldReconnect && this.reconnectAttempts < this.maxReconnectAttempts) {
        const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts);
        this.reconnectAttempts++;
        setTimeout(() => this.connect(), delay);
      }
    };

    this.ws.onerror = (event: Event) => {
      this.emit("error", event);
    };
  }

  disconnect(): void {
    this.shouldReconnect = false;
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
