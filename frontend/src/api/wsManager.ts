import { api } from "./client";
import { WebSocketManager } from "./websocket";

/** Module-level WebSocket singleton: connect once, survive component mount/unmount cycles */
export const ws = new WebSocketManager(api.getWebSocketUrl("/sdapi/v2/ws"));

let wsConnected = false;
let wsStarted = false;

export function isWsConnected() {
  return wsConnected;
}

export function ensureWs() {
  if (wsStarted) return;
  wsStarted = true;
  ws.on("open", () => { wsConnected = true; });
  ws.on("close", () => { wsConnected = false; });
  ws.connect();
}
