/**
 * Compute the number of latent sections for a FramePack generation.
 * Ported from Python's `get_latent_paddings()` in `modules/framepack/framepack_worker.py`.
 */
export function computeSectionCount(
  fps: number,
  duration: number,
  latentWindowSize: number,
  variant: string,
  interpolate: number,
): number {
  try {
    const realFps = fps / (interpolate + 1);
    const raw = (duration * realFps) / (latentWindowSize * 4);
    if (variant === "forward-only") {
      return Math.max(Math.round(raw), 1);
    }
    return Math.max(Math.floor(raw), 1);
  } catch {
    return 1;
  }
}
