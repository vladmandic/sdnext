import { useEffect, useMemo, useSyncExternalStore } from "react";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Toaster } from "@/components/ui/sonner";
import { AppShell } from "@/components/layout/AppShell";
import { useUiStore } from "@/stores/uiStore";
import { useConnectionStore } from "@/stores/connectionStore";
import { api } from "@/api/client";
import { contrastText } from "@/lib/utils";
import "./App.css";

function getSystemDark() {
  return window.matchMedia("(prefers-color-scheme: dark)").matches;
}

function subscribeSystemTheme(cb: () => void) {
  const mq = window.matchMedia("(prefers-color-scheme: dark)");
  mq.addEventListener("change", cb);
  return () => mq.removeEventListener("change", cb);
}

function useResolvedTheme(): "dark" | "light" {
  const colorMode = useUiStore((s) => s.colorMode);
  const systemDark = useSyncExternalStore(subscribeSystemTheme, getSystemDark);
  if (colorMode === "system") return systemDark ? "dark" : "light";
  return colorMode;
}

function App() {
  const cornerStyle = useUiStore((s) => s.cornerStyle);
  const accentColor = useUiStore((s) => s.accentColor);
  const borderRadius = useUiStore((s) => s.borderRadius);
  const uiScale = useUiStore((s) => s.uiScale);
  const resolvedTheme = useResolvedTheme();

  // Bootstrap stored backend connection before queries fire
  useEffect(() => {
    const { backendUrl, username, password } = useConnectionStore.getState();
    if (backendUrl) api.setBaseUrl(backendUrl);
    if (username && password) api.setAuth(username, password);
  }, []);

  // Corner style
  useEffect(() => {
    document.documentElement.dataset.corners = cornerStyle;
  }, [cornerStyle]);

  // Color mode
  useEffect(() => {
    if (resolvedTheme === "light") {
      document.documentElement.dataset.theme = "light";
    } else {
      delete document.documentElement.dataset.theme;
    }
  }, [resolvedTheme]);

  // Accent color — set CSS custom properties for primary/ring/sidebar-primary/chart-1
  const accentProps = useMemo(() => {
    const fg = contrastText(accentColor);
    return {
      "--primary": accentColor,
      "--primary-foreground": fg,
      "--ring": accentColor,
      "--sidebar-primary": accentColor,
      "--sidebar-primary-foreground": fg,
      "--sidebar-ring": accentColor,
      "--chart-1": accentColor,
    } as Record<string, string>;
  }, [accentColor]);

  useEffect(() => {
    const el = document.documentElement;
    for (const [prop, val] of Object.entries(accentProps)) {
      el.style.setProperty(prop, val);
    }
    return () => {
      for (const prop of Object.keys(accentProps)) {
        el.style.removeProperty(prop);
      }
    };
  }, [accentProps]);

  // UI scale
  useEffect(() => {
    document.documentElement.style.fontSize = `${uiScale}px`;
    return () => { document.documentElement.style.fontSize = ""; };
  }, [uiScale]);

  // Border radius
  useEffect(() => {
    document.documentElement.style.setProperty("--radius", `${borderRadius}rem`);
    return () => { document.documentElement.style.removeProperty("--radius"); };
  }, [borderRadius]);

  return (
    <TooltipProvider delayDuration={300} skipDelayDuration={300}>
      <AppShell />
      <Toaster position="bottom-right" richColors closeButton theme={resolvedTheme} />
    </TooltipProvider>
  );
}

export default App;
