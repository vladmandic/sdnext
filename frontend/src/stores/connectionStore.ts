import { create } from "zustand";
import { persist } from "zustand/middleware";

interface BackendConnection {
  id: string;
  name: string;
  url: string;
  auth?: { username: string; password: string };
  lastConnected?: number;
}

interface ConnectionState {
  connections: BackendConnection[];
  activeConnectionId: string | null;
  connectionStatus: "connected" | "disconnected" | "connecting" | "error";
  error: string | null;

  addConnection: (connection: BackendConnection) => void;
  removeConnection: (id: string) => void;
  setActiveConnection: (id: string | null) => void;
  setConnectionStatus: (status: ConnectionState["connectionStatus"]) => void;
  setError: (error: string | null) => void;
  getActiveConnection: () => BackendConnection | undefined;
}

export const useConnectionStore = create<ConnectionState>()(
  persist(
    (set, get) => ({
      connections: [],
      activeConnectionId: null,
      connectionStatus: "disconnected",
      error: null,

      addConnection: (connection) =>
        set((state) => ({
          connections: [...state.connections.filter((c) => c.id !== connection.id), connection],
        })),

      removeConnection: (id) =>
        set((state) => ({
          connections: state.connections.filter((c) => c.id !== id),
          activeConnectionId: state.activeConnectionId === id ? null : state.activeConnectionId,
        })),

      setActiveConnection: (id) => set({ activeConnectionId: id }),

      setConnectionStatus: (status) => set({ connectionStatus: status }),

      setError: (error) => set({ error }),

      getActiveConnection: () => {
        const state = get();
        return state.connections.find((c) => c.id === state.activeConnectionId);
      },
    }),
    { name: "sdnext-connections" },
  ),
);
