import { debug } from './logger';

export const allTimers: [string, number][] = [];

export async function timer(name: string, elapsed: number): Promise<void> {
  allTimers.push([name, Math.round(elapsed)]);
}
window.timer = timer;

export async function logTimers(): Promise<void> {
  // allTimers.sort((a, b) => b[1] - a[1]);
  const filteredTimers = allTimers.filter((t) => t[1] > 100);
  const objTimers: Record<string, number> = {};
  for (const [name, elapsed] of filteredTimers) objTimers[name] = elapsed;
  debug('startupTimers', objTimers);
  // xhrPost(`${window.api}/log`, { debug: JSON.stringify(objTimers) });
}
