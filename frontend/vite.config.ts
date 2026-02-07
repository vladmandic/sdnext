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
      "/sdapi/v1/ws": { target: "http://localhost:7860", ws: true },
      "/sdapi": "http://localhost:7860",
      "/internal": "http://localhost:7860",
      "/file": "http://localhost:7860",
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
  },
});
