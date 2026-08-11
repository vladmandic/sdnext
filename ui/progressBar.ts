import { log, debug, error, xhrPost } from './logger';
import { gradioApp } from './script';
import { sendNotification } from './notification';

let lastState: any = {};
let refreshInterval = 10000;
const progressTimeout = 180;
const startTimeout = 5;

export function setRefreshInterval() {
  refreshInterval = window.opts.live_preview_refresh_period || 500;
  log('refreshInterval', { visibile: document.visibilityState, interval: refreshInterval });
  document.addEventListener('visibilitychange', () => {
    if (window.opts.live_preview_require_focus !== false && document.hidden) refreshInterval = Math.max(2500, window.opts.live_preview_refresh_period || 1000);
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
  const job = res?.textinfo || res?.job || ''; // stage label when the backend reports one, job name otherwise
  let perc: string;
  let eta = '';
  if (job === 'VAE') perc = 'Decode';
  else {
    perc = res && (progress > 0) && (progress < 1) ? `${Math.round(100.0 * progress)}% ` : '';
    let sec = res?.eta || 0;
    if (res?.paused) eta = 'Paused';
    else if (res?.completed || (progress > 0.99)) eta = 'Finishing';
    else if (job.startsWith('VAE') || job.startsWith('Load')) eta = '';
    else if (sec === 0) eta = 'Start';
    else {
      const min = Math.floor(sec / 60);
      sec %= 60;
      eta = min > 0 ? `${Math.round(min)}m ${Math.round(sec)}s` : `${Math.round(sec)}s`;
    }
  }
  const elPerf = document.getElementById('control-performance');
  let hint = '';
  if (elPerf && res) {
    const jobTxt = res.job && res.job !== '' ? ` | Job ${res.job}${res.textinfo ? `: ${res.textinfo}` : ''}` : '';
    const batchTxt = res.batch > 0 ? ` | Batch ${res.batch}/${res.batches}` : '';
    const stateTxt = res.queued ? 'Queued' : res.paused ? 'Paused' : res.completed ? 'Completed' : res.active ? 'Active' : 'Idle'; // eslint-disable-line no-nested-ternary
    const stepsTxt = res.step > 0 ? ` | Step ${res.step}/${res.steps}` : '';
    const progressTxt = res.progress > 0 ? ` | Progress ${Math.round(100.0 * res.progress)}%` : '';
    const etaTxt = res.eta > 0 ? ` | ETA ${formatTime(res.eta)}` : '';
    const previewTxt = res.id_live_preview > 0 ? ` | Preview ${res.id_live_preview}` : '';
    const elapsedTxt = res.job_time > 0 ? ` | Elapsed ${formatTime((Date.now() / 1000) - res.job_time)}` : '';
    const startedTxt = res.job_time > 0 ? ` | Started ${new Date(res.job_time * 1000).toLocaleTimeString()}` : '';
    hint = `⏱ State ${stateTxt} ${jobTxt} ${startedTxt} ${elapsedTxt} ${batchTxt} ${progressTxt} ${stepsTxt} ${etaTxt} ${previewTxt}`.replaceAll('  ', ' ').trim();
    elPerf.innerHTML = `<p>${hint}`;
  }
  document.title = `SD.Next ${perc}`;
  for (const elId of elements) {
    const el = document.getElementById(elId);
    if (!el) continue;
    const jobLabel = (res ? `${job} ${perc}${eta}` : 'Generate').trim();
    el.innerText = jobLabel;
    el.title = hint.length > 0 ? hint : jobLabel;
    if (!window.waitForUiReady) {
      const gradient = perc !== '' ? perc : '100%';
      if (jobLabel === 'Generate') el.style.background = 'var(--primary-500)';
      else if (jobLabel.endsWith('Decode')) continue;
      else if (jobLabel.endsWith('Start') || jobLabel.endsWith('Finishing')) el.style.background = 'var(--primary-800)';
      else if (res && progress > 0 && progress < 1) el.style.background = `linear-gradient(to right, var(--primary-500) 0%, var(--primary-800) ${gradient}, var(--neutral-700) ${gradient})`;
      else el.style.background = 'var(--primary-500)';
    }
  }
  const el = document.getElementById('control-performance');
  if (el && res) {
    const jobTxt = res.job && res.job !== '' ? ` | Job ${res.job}${res.textinfo ? `: ${res.textinfo}` : ''}` : '';
    const batchTxt = res.batch > 0 ? ` | Batch ${res.batch}/${res.batches}` : '';
    const stateTxt = res.queued ? 'Queued' : res.paused ? 'Paused' : res.completed ? 'Completed' : res.active ? 'Active' : 'Idle'; // eslint-disable-line no-nested-ternary
    const stepsTxt = res.step > 0 ? ` | Step ${res.step}/${res.steps}` : '';
    const progressTxt = res.progress > 0 ? ` | Progress ${Math.round(100.0 * res.progress)}%` : '';
    const etaTxt = res.eta > 0 ? ` | ETA ${formatTime(res.eta)}` : '';
    const previewTxt = res.id_live_preview > 0 ? ` | Preview ${res.id_live_preview}` : '';
    const elapsedTxt = res.job_time > 0 ? ` | Elapsed ${formatTime((Date.now() / 1000) - res.job_time)}` : '';
    const startedTxt = res.job_time > 0 ? ` | Started ${new Date(res.job_time * 1000).toLocaleTimeString()}` : '';
    el.innerHTML = `<p>⏱ State ${stateTxt} ${jobTxt} ${startedTxt} ${elapsedTxt} ${batchTxt} ${progressTxt} ${stepsTxt} ${etaTxt} ${previewTxt}</p>`.replaceAll('  ', ' ').trim();
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
      const anchored = livePreview.parentElement === parentGallery;
      if (anchored) {
        // parentGallery.style.minHeight = `min(82vh, ${img.naturalHeight}px)`;
        // parentGallery.style.maxHeight = `min(82vh, ${img.naturalHeight}px)`;
        parentGallery.style.overflow = 'hidden';
      }
    };
  };

  const removeLivePreview = (useImage = false) => {
    debug('taskEnd:', id_task);
    localStorage.removeItem('task');
    setProgress();
    const footers = Array.from<any>(gradioApp().querySelectorAll('.gallery_footer'));
    for (const footer of footers) footer.style.display = 'flex'; // restore all footers
    const galleries = Array.from<any>(gradioApp().querySelectorAll('.gallery_main'));
    for (const gallery of galleries) gallery.style.display = 'flex'; // remove all galleries
    try {
      if (parentGallery && livePreview) {
        if (useImage) {
          const previewImg = gradioApp().querySelector('#livePreviewImage');
          const galleryImg = parentGallery.querySelector('img');
          if (previewImg?.src && galleryImg) galleryImg.src = previewImg.src; // copy preview to gallery if everything is ok
        }
        parentGallery.removeChild(livePreview);
      }
      if (parentGallery) {
        // parentGallery.style.minHeight = 'unset';
        // parentGallery.style.maxHeight = 'unset';
        parentGallery.style.overflow = 'unset';
      }
    } catch { /* ignore */ }
    checkPaused(true);
    sendNotification();
    if (atEnd) atEnd();
  };

  const previewVisible = () => {
    try {
      return !galleryEl?.closest('.section')?.classList.contains('minimize');
    } catch {
      return true;
    }
  };

  const onProgressDataHandler = async (res, caller) => {
    if (res?.debug) debug('progress:', { start: dateStart, res });
    lastState = res;
    const elapsedFromStart = (Date.now() - dateStart) / 1000;
    hasStarted = hasStarted || res.active;
    if (res.completed || (!res.active && (hasStarted || once))) {
      debug('progress', { end: res, reason: res.completed ? 'completed' : 'inactive' });
      const hidden = document.hidden || !previewVisible();
      if (!res.paused) removeLivePreview(!hidden); // only abort if not paused
      return;
    }
    if (elapsedFromStart > progressTimeout && !res.queued && res.progress === prevProgress) {
      debug('progress', { end: res, reason: 'progressTimeout' });
      if (!res.paused) removeLivePreview(false); // only abort if not paused
      return;
    }
    if (elapsedFromStart > startTimeout && !res.queued && !res.active) {
      debug('progress', { end: res, reason: 'startTimeout' });
      if (!res.paused) removeLivePreview(false); // only abort if not paused
      return;
    }
    if (res.progress !== prevProgress) {
      dateStart = Date.now();
      prevProgress = res.progress;
    }
    setProgress(res);
    if (res.live_preview && !livePreview) initLivePreview();
    let id_live_preview = res.id_live_preview;
    if (res.live_preview && galleryEl) {
      if (img.src !== res.live_preview) img.src = res.live_preview;
      id_live_preview = res.id_live_preview;
    }
    if (onProgress) onProgress(res);
    // timeout should be random +/- 20% of max(window.opts.live_preview_refresh_period || 500, 500))
    let timeout = Math.max(window.opts.live_preview_refresh_period || 500, 500);
    timeout += (Math.random() * 0.4 - 0.2) * timeout;
    setTimeout(() => caller(id_task, id_live_preview), timeout);
  };

  const onProgressErrorHandler = (err) => {
    error('progress', { error: err });
    removeLivePreview(false);
  };

  const startLivePreview = (taskId: string, id_live_preview: number) => {
    const hidden = document.hidden || !previewVisible();
    let request_id = id_live_preview;
    if (hidden) {
      if (!window.opts.live_preview_require_focus) request_id = id_live_preview;
    } else if (window.opts.live_preview_refresh_period === 0) {
      request_id = -1;
    }
    // eslint-disable-next-line @typescript-eslint/no-use-before-define
    xhrPost('./internal/progress', { id_task, id_live_preview: request_id }, onLivePreviewHandler, onProgressErrorHandler, false, 30000); // poll for preview
  };

  const startProgress = (taskId: string, id_live_preview: number) => {
    // eslint-disable-next-line @typescript-eslint/no-use-before-define
    xhrPost('./internal/progress', { id_task, id_live_preview: -1 }, onProgressHandler, onProgressErrorHandler, false, 30000); // poll for progress
  };

  const onProgressHandler = (res) => onProgressDataHandler(res, startProgress);
  const onLivePreviewHandler = (res) => onProgressDataHandler(res, startLivePreview);

  debug('progress', { start: dateStart });
  startLivePreview(id_task, 0);
  startProgress(id_task, -1);
}

window.checkPaused = checkPaused;
window.requestInterrupt = requestInterrupt;
window.randomId = randomId;
window.requestProgress = requestProgress;
