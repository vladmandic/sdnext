import path from "path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { VitePWA } from "vite-plugin-pwa";

export default defineConfig(({ mode }) => {
  return {
  base: mode === "production" ? "/ui/" : "/",
  plugins: [
    react(),
    tailwindcss(),
    VitePWA({
      registerType: "autoUpdate",
      devOptions: {
        enabled: true,
        type: "module",
      },
      workbox: {
        // Only precache the app shell, not API calls or dynamic content
        globPatterns: ["**/*.{js,css,html,svg,png,woff2}"],
        // Don't cache API responses or uploaded files
        navigateFallback: mode === "production" ? "/ui/index.html" : "/index.html",
        navigateFallbackDenylist: [/^\/sdapi/, /^\/internal/, /^\/file/],
        runtimeCaching: [
          {
            // Cache font files
            urlPattern: /\.(?:woff2?|ttf|otf|eot)$/,
            handler: "CacheFirst",
            options: { cacheName: "fonts", expiration: { maxEntries: 20, maxAgeSeconds: 365 * 24 * 60 * 60 } },
          },
        ],
      },
      manifest: {
        name: "SD.Next React",
        short_name: "SD.Next React",
        description: "AI Image & Video Generation",
        theme_color: "#0a0a0a",
        background_color: "#0a0a0a",
        display: "standalone",
        orientation: "any",
        scope: mode === "production" ? "/ui/" : "/",
        start_url: mode === "production" ? "/ui/" : "/",
        icons: [
          { src: "pwa-192.png", sizes: "192x192", type: "image/png" },
          { src: "pwa-512.png", sizes: "512x512", type: "image/png" },
          { src: "pwa-512.png", sizes: "512x512", type: "image/png", purpose: "maskable" },
          { src: "favicon.svg", sizes: "any", type: "image/svg+xml" },
        ],
      },
    }),
  ],
  resolve: {
    alias: { "@": path.resolve(__dirname, "./src") },
  },
  server: {
    port: 5173,
    allowedHosts: true,
    proxy: {
      "/sdapi/v1/ws": { target: "http://localhost:7860", ws: true },
      "/sdapi/v1/browser/files": { target: "http://localhost:7860", ws: true },
      "/sdapi": "http://localhost:7860",
      "/internal": "http://localhost:7860",
      "/file": "http://localhost:7860",
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
  },
};
});
