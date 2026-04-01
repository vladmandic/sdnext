let lastState = {};
let refreshInterval = 10000;

function setRefreshInterval() {
  refreshInterval = opts.live_preview_refresh_period || 500;
  log('refreshInterval', document.visibilityState, refreshInterval);
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) refreshInterval = Math.max(2500, opts.live_preview_refresh_period || 1000);
    else refreshInterval = opts.live_preview_refresh_period || 1000;
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

function checkPaused(state) {
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

function setProgress(res) {
  const elements = ['txt2img_generate', 'img2img_generate', 'extras_generate', 'control_generate', 'video_generate', 'framepack_generate'];
  const progress = res?.progress || 0;
  const job = res?.job || '';
  let perc = '';
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

function requestInterrupt() {
  setProgress();
}

function randomId() {
  return `task(${Math.random().toString(36).slice(2, 7)}${Math.random().toString(36).slice(2, 7)}${Math.random().toString(36).slice(2, 7)})`;
}

// starts sending progress requests to "/internal/progress" uri, creating progressbar above progressbarContainer element and preview inside gallery element
// Cleans up all created stuff when the task is over and calls atEnd. calls onProgress every time there is a progress update
function requestProgress(id_task, progressEl, galleryEl, atEnd = null, onProgress = null, once = false) {
  localStorage.setItem('task', id_task);
  let hasStarted = false;
  let dateStart = new Date();
  let prevProgress = null;
  const parentGallery = galleryEl ? galleryEl.parentNode : null;
  let livePreview;
  let img;

  const initLivePreview = () => {
    if (!parentGallery) return;
    const footers = Array.from(gradioApp().querySelectorAll('.gallery_footer'));
    for (const footer of footers) {
      if (footer.id !== 'gallery_footer') footer.style.display = 'none'; // remove all footers
    }
    const galleries = Array.from(gradioApp().querySelectorAll('.gallery_main'));
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
      parentGallery.style.minHeight = `min(82vh, ${img.naturalWidth}px)`;
      parentGallery.style.maxHeight = `min(82vh, ${img.naturalHeight}px)`;
      parentGallery.style.overflow = 'hidden';
    };
  };

  const done = () => {
    debug('taskEnd:', id_task);
    localStorage.removeItem('task');
    setProgress();
    const footers = Array.from(gradioApp().querySelectorAll('.gallery_footer'));
    for (const footer of footers) footer.style.display = 'flex'; // restore all footers
    const galleries = Array.from(gradioApp().querySelectorAll('.gallery_main'));
    for (const gallery of galleries) gallery.style.display = 'flex'; // remove all galleries
    try {
      if (parentGallery && livePreview) {
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

  const start = (id_task, id_live_preview) => { // eslint-disable-line no-shadow
    if (opts.live_preview_refresh_period === 0) return;
    const request_id = document.hidden ? -1 : id_live_preview;

    const onProgressHandler = (res) => {
      if (res?.debug) debug('livePreview:', dateStart, request_id, res);
      lastState = res;
      const elapsedFromStart = (new Date() - dateStart) / 1000;
      hasStarted |= res.active;
      if (res.completed || (!res.active && (hasStarted || once)) || (elapsedFromStart > 120 && !res.queued && res.progress === prevProgress)) {
        debug('livePreview end:', res);
        done();
        return;
      }
      if (res.progress !== prevProgress) {
        dateStart = new Date();
        prevProgress = res.progress;
      }
      setProgress(res);
      if (res.live_preview && !livePreview) initLivePreview();
      if (res.live_preview && galleryEl) {
        if (img.src !== res.live_preview) img.src = res.live_preview;
        id_live_preview = res.id_live_preview;
      }
      if (onProgress) onProgress(res);
      setTimeout(() => start(id_task, id_live_preview), opts.live_preview_refresh_period || 500);
    };

    const onProgressErrorHandler = (err) => {
      error(`livePreview: ${err}`);
      done();
    };

    xhrPost('./internal/progress', { id_task, id_live_preview: request_id }, onProgressHandler, onProgressErrorHandler, false, 30000);
  };
  debug('livePreview start:', dateStart);
  start(id_task, 0);
}
