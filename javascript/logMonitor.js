let logMonitorEl = null;
let logMonitorStatus = true;
let logWarnings = 0;
let logErrors = 0;
let logConnected = false;

function dateToStr(ts) {
  const dt = new Date(1000 * ts);
  const year = dt.getFullYear();
  const mo = String(dt.getMonth() + 1).padStart(2, '0');
  const day = String(dt.getDate()).padStart(2, '0');
  const hour = String(dt.getHours()).padStart(2, '0');
  const min = String(dt.getMinutes()).padStart(2, '0');
  const sec = String(dt.getSeconds()).padStart(2, '0');
  const ms = String(dt.getMilliseconds()).padStart(3, '0');
  const s = `${year}-${mo}-${day} ${hour}:${min}:${sec}.${ms}`;
  return s;
}

async function logMonitor() {
  const addLogLine = (line) => {
    try {
      const l = JSON.parse(line.replaceAll('\n', ' '));
      const row = document.createElement('tr');
      // row.style = 'padding: 10px; margin: 0;';
      const level = `<td style="color: var(--color-${l.level.toLowerCase()})">${l.level}</td>`;
      if (l.level === 'WARNING') logWarnings++;
      if (l.level === 'ERROR') logErrors++;
      const module = `<td style="color: var(--var(--neutral-400))">${l.module}</td>`;
      row.innerHTML = `<td>${dateToStr(l.created)}</td>${level}<td>${l.facility}</td>${module}<td>${l.msg}</td>`;
      logMonitorEl.appendChild(row);
    } catch (e) {
      error(`logMonitor: ${line}`);
    }
  };

  const cleanupLog = (atBottom) => {
    while (logMonitorEl.childElementCount > 100) logMonitorEl.removeChild(logMonitorEl.firstChild);
    if (atBottom) logMonitorEl.scrollTop = logMonitorEl.scrollHeight;
    else logMonitorEl.parentElement.style = 'border-bottom: 2px solid var(--highlight-color);';
    const elWarn = document.getElementById('logWarnings');
    const elErr = document.getElementById('logErrors');
    const modenUIBtn = document.getElementById('btn_console');
    if (elWarn) elWarn.innerText = logWarnings;
    if (elErr) elErr.innerText = logErrors;
    if (modenUIBtn) modenUIBtn.setAttribute('error-count', logErrors > 0 ? logErrors : '');
  };

  const txtGallery = document.getElementById('txt2img_gallery');
  if (txtGallery) txtGallery.style.height = opts.logmonitor_show ? '50vh' : '55vh';
  const imgGallery = document.getElementById('img2img_gallery');
  if (imgGallery) imgGallery.style.height = opts.logmonitor_show ? '50vh' : '55vh';

  if (!opts.logmonitor_show) {
    Array.from(document.getElementsByClassName('log-monitor')).forEach((el) => el.style.display = 'none');
    return;
  }

  if (logMonitorStatus) setTimeout(logMonitor, opts.logmonitor_refresh_period);
  else setTimeout(logMonitor, 10 * 1000); // on failure try to reconnect every 10sec

  logMonitorStatus = false;
  if (!logMonitorEl) {
    logMonitorEl = document.getElementById('logMonitorData');
    logMonitorEl.onscrollend = () => {
      const atBottom = logMonitorEl.scrollHeight <= (logMonitorEl.scrollTop + logMonitorEl.clientHeight);
      if (atBottom) logMonitorEl.parentElement.style = '';
    };
  }
  if (!logMonitorEl) return;
  const atBottom = logMonitorEl.scrollHeight <= (logMonitorEl.scrollTop + logMonitorEl.clientHeight);
  try {
    const res = await fetch(`${window.api}/log?clear=True`);
    if (res?.ok) {
      logMonitorStatus = true;
      const lines = await res.json();
      if (logMonitorEl && lines?.length > 0) logMonitorEl.parentElement.parentElement.style.display = opts.logmonitor_show ? 'block' : 'none';
      for (const line of lines) addLogLine(line);
      if (!logConnected) {
        logConnected = true;
        monitorConnection();
        xhrPost(`${window.api}/log`, { debug: 'connected' });
      }
    } else {
      logConnected = false;
      logErrors++;
      addLogLine(`{ "created": ${Date.now()}, "level":"ERROR", "module":"logMonitor", "facility":"ui", "msg":"Failed to fetch log: ${res?.status} ${res?.statusText}" }`);
    }
    cleanupLog(atBottom);
  } catch (err) {
    logConnected = false;
    logErrors++;
    addLogLine(`{ "created": ${Date.now()}, "level":"ERROR", "module":"logMonitor", "facility":"ui", "msg":"Failed to fetch log: server unreachable" }`);
    cleanupLog(atBottom);
  }
}

async function initLogMonitor() {
  const el = document.getElementsByTagName('footer')[0];
  if (!el) return;
  el.classList.add('log-monitor');
  const ui_disabled = Array.isArray(window.opts.ui_disabled) ? window.opts.ui_disabled : [];
  if (ui_disabled.includes('logs')) return;
  el.innerHTML = `
    <table id="logMonitor" style="width: 100%;">
      <thead style="display: block; text-align: left; border-bottom: solid 1px var(--button-primary-border-color)">
        <tr>
          <th style="width: 144px">Time</th>
          <th>Level</th>
          <th style="width: 0"></th>
          <th style="width: 154px">Module</th>
          <th>Message</th>
          <th style="position: absolute; right: 7em">Warnings <span id="logWarnings">0</span></th>
          <th style="position: absolute; right: 1em">Errors <span id="logErrors">0</span></th>
        </tr>
      </thead>
      <tbody id="logMonitorData" style="white-space: nowrap; height: 10vh; width: 100vw; display: block; overflow-x: hidden; overflow-y: scroll; color: var(--neutral-400)">
      </tbody>
    </table>
  `;
  el.style.display = 'none';
  fetch(`${window.api}/start?agent=${encodeURI(navigator.userAgent)}`);
  logMonitor();
  log('initLogMonitor');
}
