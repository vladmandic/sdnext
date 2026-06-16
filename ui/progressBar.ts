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
    // log('refreshInterval', document.visibilityState, refreshInterval);
  });
}

function pad2(x) {
  return x < 10 ? `0${x}` : x;
}

function formatTime(secs) {
  if (secs > 3600) return `${pad2(Math.floor(secs / 60 / 60))}:${pad2(Math.floor(secs / 60) % 60)}:${pad2(Math.floor(secs) % 60)}`;
  if (secs > 60) return `${pad2(Math.floor(secs / 60))}:${pad2(Math.floor(secs) % 60)}`;
  return `${Math.floor(secs)}s`;
}

export function checkPaused(state) {
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

// starts sending progress requests to "/internal/progress" uri, creating progressbar above progressbarContainer element and preview inside gallery element
// Cleans up all created stuff when the task is over and calls atEnd. calls onProgress every time there is a progress update
export function requestProgress(id_task = 'undefined', progressEl = null, galleryEl = null, atEnd = null, onProgress = null, once = false) {
  if (id_task) localStorage.setItem('task', id_task);
  let hasStarted = false;
  let dateStart = Date.now();
  let prevProgress: any = null;
  const parentGallery = galleryEl ? galleryEl.parentNode : null;
  let livePreview: HTMLElement | undefined;
  let img: HTMLImageElement;

  const initLivePreview = () => {
    if (!parentGallery) return;
    const footers = Array.from<any>(gradioApp().querySelectorAll('.gallery_footer'));
    for (const footer of footers) {
      if (footer.id !== 'gallery_footer') footer.style.display = 'none'; // remove all footers
    }
    const galleries = Array.from<any>(gradioApp().querySelectorAll('.gallery_main'));
    for (const gallery of galleries) {
      if (gallery.id !== 'gallery_gallery') gallery.style.display = 'none'; // remove all footers
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
    const footers = Array.from<any>(gradioApp().querySelectorAll('.gallery_footer'));
    for (const footer of footers) footer.style.display = 'flex'; // restore all footers
    const galleries = Array.from<any>(gradioApp().querySelectorAll('.gallery_main'));
    for (const gallery of galleries) gallery.style.display = 'flex'; // remove all galleries
    try {
      if (parentGallery && livePreview) {
        if (ok) {
          const previewImg = gradioApp().querySelector('#livePreviewImage');
          const galleryImg = gradioApp().querySelector('#control_gallery img');
          if (previewImg?.src && galleryImg) galleryImg.src = previewImg.src; // copy preview to gallery if everything is ok
        }
        parentGallery.removeChild(livePreview);
        parentGallery.style.minHeight = 'unset';
        parentGallery.style.maxHeight = 'unset';
        parentGallery.style.overflow = 'unset';
      }
    } catch { /* ignore */ }
    checkPaused(true);
    sendNotification();
    if (atEnd) atEnd();
  };

  const onProgressHandler = (res) => {
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
    if (res.live_preview && !livePreview) initLivePreview();
    if (res.live_preview && galleryEl) {
      if (img.src !== res.live_preview) img.src = res.live_preview;
    }
    if (onProgress) onProgress(res);
  };

  const onProgressErrorHandler = (err) => {
    error('progress', { error: err });
    removeLivePreview(false);
  };

  const startHttpPolling = (taskId: string, id_live_preview: number) => {
    if (window.opts.live_preview_refresh_period === 0) return;
    const request_id = document.hidden ? -1 : id_live_preview;
    const wrappedHandler = (res) => {
      onProgressHandler(res);
      if (res.completed || (!res.active && (hasStarted || once))) return;
      if (!res.paused) {
        setTimeout(() => startHttpPolling(taskId, res.id_live_preview || 0), window.opts.live_preview_refresh_period || 500);
      }
    };
    xhrPost('./internal/progress', { id_task: taskId, id_live_preview: request_id }, wrappedHandler, onProgressErrorHandler, false, 30000);
  };

  const startLivePreviewWebSocket = () => {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${proto}//${location.host}/sdapi/v1/preview`);
    let wsFailed = false;
    let revokeUrl: string | undefined;

    const sendVisibility = () => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'visibility', visible: !document.hidden }));
      }
    };

    const cleanup = (ok: boolean) => {
      document.removeEventListener('visibilitychange', sendVisibility);
      if (revokeUrl) { URL.revokeObjectURL(revokeUrl); revokeUrl = undefined; }
      ws.close();
      if (!ok && !wsFailed) {
        wsFailed = true;
        debug('progress', 'ws fallback to polling');
        startHttpPolling(id_task, 0);
      }
    };

    ws.onopen = () => {
      sendVisibility();
      debug('progress', 'ws connected');
    };

    ws.onmessage = (event) => {
      if (event.data instanceof Blob) {
        if (!livePreview) initLivePreview();
        const url = URL.createObjectURL(event.data);
        if (revokeUrl) URL.revokeObjectURL(revokeUrl);
        revokeUrl = url;
        img.src = url;
        if (onProgress) onProgress(lastState);
      } else if (typeof event.data === 'string') {
        try {
          const res = JSON.parse(event.data);
          if (res.completed) {
            onProgressHandler(res);
            cleanup(true);
            removeLivePreview(true);
            return;
          }
          onProgressHandler(res);
        } catch (e) {
          error('progress ws', { error: e });
        }
      }
    };

    ws.onerror = () => { cleanup(false); };

    ws.onclose = (event) => {
      if (event.code !== 1000) cleanup(false);
    };

    document.addEventListener('visibilitychange', sendVisibility);

    const fallbackTimer = setTimeout(() => {
      if (ws.readyState !== WebSocket.OPEN) {
        wsFailed = true;
        debug('progress', 'ws timeout, fallback to polling');
        ws.close();
        startHttpPolling(id_task, 0);
      }
    }, 1000);

    ws.addEventListener('open', () => clearTimeout(fallbackTimer));
  };

  debug('progress', { start: dateStart });
  const transport = window.opts.live_preview_transport || 'Polling';
  if (transport === 'WebSocket') {
    startLivePreviewWebSocket();
  } else {
    startHttpPolling(id_task, 0);
  }
}

window.checkPaused = checkPaused;
window.requestInterrupt = requestInterrupt;
window.randomId = randomId;
window.requestProgress = requestProgress;
