import { create } from "zustand";
import { persist } from "zustand/middleware";

interface ConnectionState {
  /** Empty string means use window.location.origin (default) */
  backendUrl: string;
  username: string;
  password: string;
  setBackendUrl: (url: string) => void;
  setAuth: (username: string, password: string) => void;
  clearAuth: () => void;
  reset: () => void;
}

export const useConnectionStore = create<ConnectionState>()(
  persist(
    (set) => ({
      backendUrl: "",
      username: "",
      password: "",

      setBackendUrl: (url) => set({ backendUrl: url }),
      setAuth: (username, password) => set({ username, password }),
      clearAuth: () => set({ username: "", password: "" }),
      reset: () => set({ backendUrl: "", username: "", password: "" }),
    }),
    { name: "sdnext-connections" },
  ),
);
