import { useEffect, useState, useCallback, useRef } from "react";
import { useTutorialStore, TUTORIAL_STEPS } from "@/stores/tutorialStore";
import { useShortcutScope } from "@/hooks/useShortcutScope";
import { useShortcut } from "@/hooks/useShortcut";
import { Button } from "@/components/ui/button";

function useTutorialAutoStart() {
  const completed = useTutorialStore((s) => s.completed);
  const active = useTutorialStore((s) => s.active);
  const start = useTutorialStore((s) => s.start);

  useEffect(() => {
    if (completed || active) return;
    const keys = Object.keys(localStorage);
    const hasPriorData = keys.some((k) => k.startsWith("sdnext-") && k !== "sdnext-tutorial");
    if (hasPriorData) return;
    const timer = setTimeout(start, 800);
    return () => clearTimeout(timer);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps
}

interface Rect { top: number; left: number; width: number; height: number }

function useTargetRect(target: string, active: boolean): { rect: Rect | null; el: HTMLElement | null } {
  const [rect, setRect] = useState<Rect | null>(null);
  const [el, setEl] = useState<HTMLElement | null>(null);
  const observerRef = useRef<ResizeObserver | null>(null);

  const measure = useCallback(() => {
    const found = document.querySelector<HTMLElement>(`[data-tour="${target}"]`);
    if (found) {
      const r = found.getBoundingClientRect();
      setRect({ top: r.top, left: r.left, width: r.width, height: r.height });
      setEl(found);
    } else {
      setRect(null);
      setEl(null);
    }
  }, [target]);

  useEffect(() => {
    if (!active) return;
    requestAnimationFrame(measure);
    const found = document.querySelector<HTMLElement>(`[data-tour="${target}"]`);
    if (found) {
      observerRef.current = new ResizeObserver(measure);
      observerRef.current.observe(found);
    }
    window.addEventListener("resize", measure);
    window.addEventListener("scroll", measure, true);
    return () => {
      observerRef.current?.disconnect();
      window.removeEventListener("resize", measure);
      window.removeEventListener("scroll", measure, true);
    };
  }, [target, active, measure]);

  return { rect: active ? rect : null, el: active ? el : null };
}

function getTooltipPosition(rect: Rect, placement: string) {
  const pad = 16;
  switch (placement) {
    case "right":
      return { top: rect.top, left: rect.left + rect.width + pad };
    case "left":
      return { top: rect.top, left: rect.left - pad };
    case "top":
      return { top: rect.top - pad, left: rect.left };
    case "bottom":
      return { top: rect.top + rect.height + pad, left: rect.left };
    default:
      return { top: rect.top, left: rect.left + rect.width + pad };
  }
}

function getTooltipTransform(placement: string) {
  switch (placement) {
    case "left":
      return "translateX(-100%)";
    case "top":
      return "translateY(-100%)";
    default:
      return undefined;
  }
}

function TutorialOverlayInner() {
  const currentStep = useTutorialStore((s) => s.currentStep);
  const next = useTutorialStore((s) => s.next);
  const back = useTutorialStore((s) => s.back);
  const skip = useTutorialStore((s) => s.skip);

  const step = TUTORIAL_STEPS[currentStep];
  const { rect, el } = useTargetRect(step.target, true);
  const isLast = currentStep === TUTORIAL_STEPS.length - 1;

  useShortcutScope("tutorial", true);
  useShortcut("tutorial-close", skip);

  useEffect(() => {
    el?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [el]);

  if (!rect) return null;

  const spotPad = 8;
  const spotR = 8;
  const spotX = rect.left - spotPad;
  const spotY = rect.top - spotPad;
  const spotW = rect.width + spotPad * 2;
  const spotH = rect.height + spotPad * 2;

  const tooltipPos = getTooltipPosition(rect, step.placement);
  const tooltipTransform = getTooltipTransform(step.placement);

  return (
    <>
      {/* Backdrop with spotlight cutout */}
      <svg
        className="fixed inset-0 z-[60]"
        width="100%"
        height="100%"
        onClick={skip}
        style={{ cursor: "default" }}
      >
        <defs>
          <mask id="tutorial-mask">
            <rect width="100%" height="100%" fill="white" />
            <rect
              x={spotX}
              y={spotY}
              width={spotW}
              height={spotH}
              rx={spotR}
              ry={spotR}
              fill="black"
            />
          </mask>
        </defs>
        <rect
          width="100%"
          height="100%"
          fill="rgba(0,0,0,0.6)"
          mask="url(#tutorial-mask)"
        />
      </svg>

      {/* Tooltip card */}
      <div
        className="fixed z-[61] w-72 rounded-lg border bg-popover p-4 text-popover-foreground shadow-lg"
        style={{
          top: tooltipPos.top,
          left: tooltipPos.left,
          transform: tooltipTransform,
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="text-3xs text-muted-foreground mb-1">
          {currentStep + 1} of {TUTORIAL_STEPS.length}
        </div>
        <h3 className="text-sm font-semibold mb-1">{step.title}</h3>
        <p className="text-xs text-muted-foreground mb-4 leading-relaxed">{step.description}</p>
        <div className="flex items-center justify-between">
          <button
            type="button"
            onClick={skip}
            className="text-xs text-muted-foreground hover:text-foreground transition-colors"
          >
            Skip
          </button>
          <div className="flex gap-2">
            {currentStep > 0 && (
              <Button variant="outline" size="sm" onClick={back} className="text-xs h-7">
                Back
              </Button>
            )}
            <Button size="sm" onClick={next} className="text-xs h-7">
              {isLast ? "Done" : "Next"}
            </Button>
          </div>
        </div>
      </div>
    </>
  );
}

export function TutorialOverlay() {
  useTutorialAutoStart();
  const active = useTutorialStore((s) => s.active);
  if (!active) return null;
  return <TutorialOverlayInner />;
}
