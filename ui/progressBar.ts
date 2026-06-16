import { log, debug, error, xhrPost } from './logger';
import { gradioApp } from './script';
import { sendNotification } from './notification';

let lastState: any = {};
let refreshInterval = 10000;
const progressTimeout = 180;
const startTimeout = 5;

export function setRefreshInterval() {
  refreshInterval = window.opts.live_preview_refresh_period || 500;
  log('refreshInterval', document.visibilityState, refreshInterval);
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) refreshInterval = Math.max(2500, window.opts.live_preview_refresh_period || 1000);
    else refreshInterval = window.opts.live_preview_refresh_period || 1000;
  });
}

function pad2(x: number) {
  return x < 10 ? `0${x}` : x;
}

function formatTime(secs: number) {
  if (secs > 3600) return `${pad2(Math.floor(secs / 60 / 60))}:${pad2(Math.floor(secs / 60) % 60)}:${pad2(Math.floor(secs) % 60)}`;
  if (secs > 60) return `${pad2(Math.floor(secs / 60))}:${pad2(Math.floor(secs) % 60)}`;
  return `${Math.floor(secs)}s`;
}

export function checkPaused(state?: boolean) {
  lastState.paused = state ? !state : !lastState.paused;
  const t_el = document.getElementById('txt2img_pause');
  const i_el = document.getElementById('img2img_pause');
  const c_el = document.getElementById('control_pause');
  const v_el = document.getElementById('video_pause');
  if (t_el) t_el.innerText = lastState.paused ? 'Resume' : 'Pause';
  if (i_el) i_el.innerText = lastState.paused ? 'Resume' : 'Pause';
  if (c_el) c_el.innerText = lastState.paused ? 'Resume' : 'Pause';
  if (v_el) v_el.innerText = lastState.paused ? 'Resume' : 'Pause';
}

export function setProgress(res?: any) {
  const elements = ['txt2img_generate', 'img2img_generate', 'extras_generate', 'control_generate', 'video_generate', 'framepack_generate'];
  const progress = res?.progress || 0;
  const job = res?.job || '';
  let perc: string;
  let eta = '';
  if (job === 'VAE') perc = 'Decode';
  else {
    perc = res && (progress > 0) && (progress < 1) ? `${Math.round(100.0 * progress)}% ` : '';
    let sec = res?.eta || 0;
    if (res?.paused) eta = 'Paused';
    else if (res?.completed || (progress > 0.99)) eta = 'Finishing';
    else if (sec === 0) eta = 'Start';
    else {
      const min = Math.floor(sec / 60);
      sec %= 60;
      eta = min > 0 ? `${Math.round(min)}m ${Math.round(sec)}s` : `${Math.round(sec)}s`;
    }
  }
  document.title = `SD.Next ${perc}`;
  for (const elId of elements) {
    const el = document.getElementById(elId);
    if (el) {
      const jobLabel = (res ? `${job} ${perc}${eta}` : 'Generate').trim();
      el.innerText = jobLabel;
      if (!window.waitForUiReady) {
        const gradient = perc !== '' ? perc : '100%';
        if (jobLabel === 'Generate') el.style.background = 'var(--primary-500)';
        else if (jobLabel.endsWith('Decode')) continue;
        else if (jobLabel.endsWith('Start') || jobLabel.endsWith('Finishing')) el.style.background = 'var(--primary-800)';
        else if (res && progress > 0 && progress < 1) el.style.background = `linear-gradient(to right, var(--primary-500) 0%, var(--primary-800) ${gradient}, var(--neutral-700) ${gradient})`;
        else el.style.background = 'var(--primary-500)';
      }
    }
  }
}

export function requestInterrupt() {
  setProgress();
}

export function randomId() {
  return `task(${Math.random().toString(36).slice(2, 7)}${Math.random().toString(36).slice(2, 7)}${Math.random().toString(36).slice(2, 7)})`;
}

function getWebSocketUrl(): string {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${proto}//${location.host}/ws/preview`;
}

export function requestProgress(id_task = 'undefined', progressEl: HTMLElement | null = null, galleryEl: HTMLElement | null = null, atEnd: (() => void) | null = null, onProgress: ((res: any) => void) | null = null, once = false) {
  if (id_task) localStorage.setItem('task', id_task);
  let hasStarted = false;
  let dateStart = Date.now();
  let prevProgress: any = null;
  const parentGallery: HTMLElement | null = galleryEl ? galleryEl.parentNode as HTMLElement : null;
  let livePreview: HTMLElement | undefined;
  let img: HTMLImageElement;
  let ws: WebSocket | null = null;
  let wsReconnect: number | undefined;
  let pollingTimer: number | undefined;
  let id_live_preview = 0;

  const initLivePreview = () => {
    if (!parentGallery) return;
    const footers = Array.from<any>(gradioApp().querySelectorAll('.gallery_footer'));
    for (const footer of footers) {
      if (footer.id !== 'gallery_footer') footer.style.display = 'none';
    }
    const galleries = Array.from<any>(gradioApp().querySelectorAll('.gallery_main'));
    for (const gallery of galleries) {
      if (gallery.id !== 'gallery_gallery') gallery.style.display = 'none';
    }

    livePreview = document.createElement('div');
    livePreview.className = 'livePreview';
    parentGallery.insertBefore(livePreview, galleryEl);
    img = new Image();
    img.id = 'livePreviewImage';
    livePreview.appendChild(img);
    img.onload = () => {
      img.style.width = `min(100%, max(${img.naturalWidth}px, 512px))`;
      parentGallery.style.minHeight = `min(82vh, ${img.naturalHeight}px)`;
      parentGallery.style.maxHeight = `min(82vh, ${img.naturalHeight}px)`;
      parentGallery.style.overflow = 'hidden';
    };
  };

  const removeLivePreview = (ok = false) => {
    debug('taskEnd:', id_task);
    localStorage.removeItem('task');
    setProgress();
    if (ws) {
      try { ws.send('end'); } catch { /* ignore */ }
      try { ws.close(); } catch { /* ignore */ }
      ws = null;
    }
    if (wsReconnect) {
      clearTimeout(wsReconnect);
      wsReconnect = undefined;
    }
    if (pollingTimer) {
      clearTimeout(pollingTimer);
      pollingTimer = undefined;
    }
    const footers = gradioApp().querySelectorAll('.gallery_footer');
    for (const footer of Array.from<any>(footers)) footer.style.display = 'flex';
    const galleries = gradioApp().querySelectorAll('.gallery_main');
    for (const gallery of Array.from<any>(galleries)) gallery.style.display = 'flex';
    if (parentGallery && livePreview && livePreview.parentNode) {
      if (ok) {
        const previewImg = gradioApp().querySelector('#livePreviewImage') as HTMLImageElement;
        const galleryImg = gradioApp().querySelector('#control_gallery img') as HTMLImageElement;
        if (previewImg?.src && galleryImg) galleryImg.src = previewImg.src;
      }
      parentGallery.removeChild(livePreview);
      parentGallery.style.minHeight = 'unset';
      parentGallery.style.maxHeight = 'unset';
      parentGallery.style.overflow = 'unset';
    }
    checkPaused(true);
    sendNotification();
    if (atEnd) atEnd();
  };

  const onWsMessage = (event: MessageEvent) => {
    try {
      const data = JSON.parse(event.data);
      if (data.type === 'preview' && data.live_preview) {
        if (!livePreview) initLivePreview();
        if (livePreview && galleryEl && img && img.src !== data.live_preview) {
          img.src = data.live_preview;
          id_live_preview = data.id_live_preview;
          lastState = { ...lastState, step: data.step, steps: data.steps, progress: data.progress, job: data.job };
          setProgress(lastState);
          if (onProgress) onProgress(lastState);
          dateStart = Date.now();
          prevProgress = data.progress;
        }
      } else if (data.type === 'progress') {
        lastState = { ...lastState, step: data.step, steps: data.steps, progress: data.progress, active: data.active, paused: data.paused, job: data.job };
        setProgress(lastState);
        if (data.progress !== prevProgress) {
          dateStart = Date.now();
          prevProgress = data.progress;
        }
        if (onProgress) onProgress(lastState);
      } else if (data.type === 'complete') {
        removeLivePreview(true);
      }
    } catch { /* ignore */ }
  };

  const connectWebSocket = () => {
    if (ws) return;
    try {
      ws = new WebSocket(getWebSocketUrl());
      ws.onmessage = onWsMessage;
      ws.onopen = () => {
        debug('ws', 'connected');
        if (wsReconnect) {
          clearTimeout(wsReconnect);
          wsReconnect = undefined;
        }
      };
      ws.onclose = () => {
        debug('ws', 'disconnected');
        ws = null;
        if (pollingTimer !== undefined) {
          wsReconnect = window.setTimeout(connectWebSocket, 3000);
        }
      };
      ws.onerror = () => {
        debug('ws', 'error');
      };
    } catch {
      wsReconnect = window.setTimeout(connectWebSocket, 3000);
    }
  };

  const pollProgress = () => {
    if (window.opts.live_preview_refresh_period === 0) return;
    const onProgressHandler = (res: any) => {
      if (res?.debug) debug('progress:', { start: dateStart, res });
      lastState = res;
      const elapsedFromStart = (Date.now() - dateStart) / 1000;
      hasStarted = hasStarted || res.active;
      if (res.completed || (!res.active && (hasStarted || once))) {
        debug('progress', { end: res, reason: res.completed ? 'completed' : 'inactive' });
        if (!res.paused) removeLivePreview(true);
        return;
      }
      if (elapsedFromStart > progressTimeout && !res.queued && res.progress === prevProgress) {
        debug('progress', { end: res, reason: 'progressTimeout' });
        if (!res.paused) removeLivePreview(false);
        return;
      }
      if (elapsedFromStart > startTimeout && !res.queued && !res.active) {
        debug('progress', { end: res, reason: 'startTimeout' });
        if (!res.paused) removeLivePreview(false);
        return;
      }
      if (res.progress !== prevProgress) {
        dateStart = Date.now();
        prevProgress = res.progress;
      }
      setProgress(res);
      if (onProgress) onProgress(res);
      pollingTimer = window.setTimeout(pollProgress, window.opts.live_preview_refresh_period || 500);
    };

    const onProgressErrorHandler = (err: any) => {
      error('progress', { error: err });
      removeLivePreview(false);
    };

    xhrPost('./internal/progress', { id_task, id_live_preview: -1 }, onProgressHandler, onProgressErrorHandler, false, 30000);
  };

  debug('progress', { start: dateStart });
  connectWebSocket();
  pollProgress();
}

window.checkPaused = checkPaused;
window.requestInterrupt = requestInterrupt;
window.randomId = randomId;
window.requestProgress = requestProgress;
