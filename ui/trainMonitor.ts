import { gradioApp } from './script';
import { randomId, requestProgress } from './progressBar';

export function startTrainMonitor(...args: unknown[]): unknown[] {
  const errorEl = gradioApp().querySelector('#train_error');
  if (errorEl instanceof HTMLElement) errorEl.innerHTML = '';
  const id = randomId();
  const onProgress = (progress: { textinfo?: string }): void => {
    const progressEl = gradioApp().getElementById('train_progress');
    if (progressEl) progressEl.innerHTML = progress.textinfo ?? '';
  };
  requestProgress(id, gradioApp().getElementById('train_gallery'), null, onProgress, false);
  const res = [...args];
  res[0] = id;
  return res;
}
