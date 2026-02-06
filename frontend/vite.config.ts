import path from "path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { "@": path.resolve(__dirname, "./src") },
  },
  server: {
    port: 5173,
    proxy: {
      "/sdapi": "http://localhost:7860",
      "/ws": { target: "ws://localhost:7860", ws: true },
      "/internal": "http://localhost:7860",
      "/file": "http://localhost:7860",
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
  },
});
