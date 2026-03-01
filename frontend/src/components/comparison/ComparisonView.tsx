import { useComparisonStore } from "@/stores/comparisonStore";
import type { ComparisonMode } from "@/stores/comparisonStore";
import { useShortcutScope } from "@/hooks/useShortcutScope";
import { useShortcut } from "@/hooks/useShortcut";
import { SideBySideMode } from "./SideBySideMode";
import { SwipeMode } from "./SwipeMode";
import { OverlayMode } from "./OverlayMode";
import { DiffMode } from "./DiffMode";
import { X, Columns2, SplitSquareHorizontal, Layers, Grid2x2, ArrowLeftRight } from "lucide-react";

const MODE_ICONS: Record<ComparisonMode, typeof Columns2> = {
  "side-by-side": Columns2,
  "swipe": SplitSquareHorizontal,
  "overlay": Layers,
  "diff": Grid2x2,
};

const MODE_LABELS: Record<ComparisonMode, string> = {
  "side-by-side": "Side by Side",
  "swipe": "Swipe",
  "overlay": "Overlay",
  "diff": "Pixel Diff",
};

const MODES: ComparisonMode[] = ["side-by-side", "swipe", "overlay", "diff"];

export function ComparisonView() {
  const { open, mode, imageA, imageB, closeComparison, setMode, swapImages } = useComparisonStore();

  useShortcutScope("comparison", open);
  useShortcut("comparison-close", closeComparison, open);
  useShortcut("comparison-side-by-side", () => setMode("side-by-side"), open);
  useShortcut("comparison-swipe", () => setMode("swipe"), open);
  useShortcut("comparison-overlay", () => setMode("overlay"), open);
  useShortcut("comparison-diff", () => setMode("diff"), open);
  useShortcut("comparison-swap", swapImages, open);

  if (!open || !imageA || !imageB) return null;

  return (
    <div className="fixed inset-0 z-50 bg-black/95 flex flex-col" onClick={closeComparison}>
      {/* Top toolbar */}
      <div
        className="flex items-center justify-between px-4 py-2 flex-shrink-0"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center gap-1">
          {MODES.map((m) => {
            const Icon = MODE_ICONS[m];
            return (
              <button
                key={m}
                onClick={() => setMode(m)}
                className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded text-2xs transition-colors ${
                  mode === m
                    ? "bg-white/20 text-white"
                    : "text-white/60 hover:text-white hover:bg-white/10"
                }`}
                title={MODE_LABELS[m]}
              >
                <Icon size={14} />
                <span className="hidden sm:inline">{MODE_LABELS[m]}</span>
              </button>
            );
          })}
        </div>

        <div className="flex items-center gap-1">
          <ToolbarButton onClick={swapImages} title="Swap A/B">
            <ArrowLeftRight size={14} />
          </ToolbarButton>
          <div className="w-px h-4 bg-white/20 mx-1" />
          <ToolbarButton onClick={closeComparison} title="Close">
            <X size={16} />
          </ToolbarButton>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 min-h-0" onClick={(e) => e.stopPropagation()}>
        {mode === "side-by-side" && <SideBySideMode imageA={imageA} imageB={imageB} />}
        {mode === "swipe" && <SwipeMode imageA={imageA} imageB={imageB} />}
        {mode === "overlay" && <OverlayMode imageA={imageA} imageB={imageB} />}
        {mode === "diff" && <DiffMode imageA={imageA} imageB={imageB} />}
      </div>

      {/* Bottom bar */}
      <div
        className="flex items-center justify-center px-4 py-1.5 flex-shrink-0 gap-4"
        onClick={(e) => e.stopPropagation()}
      >
        <span className="text-3xs text-white/40">A: {imageA.label}</span>
        <span className="text-3xs text-white/20">vs</span>
        <span className="text-3xs text-white/40">B: {imageB.label}</span>
      </div>
    </div>
  );
}

function ToolbarButton({ children, onClick, title }: { children: React.ReactNode; onClick: () => void; title: string }) {
  return (
    <button
      onClick={onClick}
      title={title}
      className="w-8 h-8 flex items-center justify-center rounded text-white/70 hover:text-white hover:bg-white/10 transition-colors"
    >
      {children}
    </button>
  );
}
