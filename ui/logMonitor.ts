import { authFetch } from './authWrap';
import { log, error, xhrPost } from './logger';
import { timer } from './timers';

interface LogLine {
  created: number;
  level: string;
  module: string;
  facility: string;
  msg: string;
}

let logMonitorEl: HTMLElement | null = null;
let logMonitorStatus = true;
let logWarnings = 0;
let logErrors = 0;
let logConnected = false;

function dateToStr(ts: number): string {
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

function htmlEscape(text: string): string {
  return text.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
}

function parseLogLine(line: string): LogLine {
  const parsed = JSON.parse(line.replaceAll('\n', ' ').replaceAll('\\', '\\\\')) as Partial<LogLine>;
  return {
    created: Number(parsed.created ?? Date.now()),
    level: String(parsed.level ?? 'INFO'),
    module: String(parsed.module ?? 'logMonitor'),
    facility: String(parsed.facility ?? 'ui'),
    msg: String(parsed.msg ?? ''),
  };
}

async function logMonitor() {
  const addLogLine = (line: string): void => {
    if (!logMonitorEl) logMonitorEl = document.getElementById('logMonitorData');
    if (!logMonitorEl) return;
    try {
      const l = parseLogLine(line);
      const row = document.createElement('tr');
      // row.style = 'padding: 10px; margin: 0;';
      const level = `<td style="color: var(--color-${l.level.toLowerCase()})">${l.level}</td>`;
      if (l.level === 'WARNING') logWarnings++;
      if (l.level === 'ERROR') logErrors++;
      const module = `<td style="color: var(--neutral-400)">${l.module}</td>`;
      const facilityText = l.facility.length > 20 ? `${l.facility.substring(0, 20)}...` : l.facility;
      const facility = l.facility !== 'sd' ? `<td>${facilityText}</td>` : '<td></td>';
      row.innerHTML = `<td>${dateToStr(l.created)}</td>${level}${facility}${module}<td>${htmlEscape(l.msg)}</td>`;
      logMonitorEl.appendChild(row);
    } catch (err) {
      error(`logMonitor: ${String(err)}\n${line}`);
    }
  };

  const cleanupLog = (atBottom: boolean): void => {
    if (!logMonitorEl) return;
    while (logMonitorEl.childElementCount > 100 && logMonitorEl.firstElementChild) {
      logMonitorEl.removeChild(logMonitorEl.firstElementChild);
    }
    if (atBottom) logMonitorEl.scrollTop = logMonitorEl.scrollHeight;
    else if (logMonitorEl.parentElement) logMonitorEl.parentElement.style.cssText = 'border-bottom: 2px solid var(--highlight-color);';
    const elWarn = document.getElementById('logWarnings');
    const elErr = document.getElementById('logErrors');
    const modenUIBtn = document.getElementById('btn_console');
    if (elWarn) elWarn.innerText = String(logWarnings);
    if (elErr) elErr.innerText = String(logErrors);
    if (modenUIBtn) modenUIBtn.setAttribute('error-count', logErrors > 0 ? String(logErrors) : '');
  };

  const txtGallery = document.getElementById('txt2img_gallery');
  if (txtGallery) txtGallery.style.height = window.opts.logmonitor_show ? '50vh' : '55vh';
  const imgGallery = document.getElementById('img2img_gallery');
  if (imgGallery) imgGallery.style.height = window.opts.logmonitor_show ? '50vh' : '55vh';

  if (!window.opts.logmonitor_show) {
    Array.from<any>(document.getElementsByClassName('log-monitor')).forEach((el) => {
      if (el instanceof HTMLElement) el.style.display = 'none';
    });
    return;
  }

  if (logMonitorStatus) setTimeout(logMonitor, window.opts.logmonitor_refresh_period);
  else setTimeout(logMonitor, 10 * 1000); // on failure try to reconnect every 10sec

  logMonitorStatus = false;
  if (!logMonitorEl) {
    logMonitorEl = document.getElementById('logMonitorData');
    if (logMonitorEl) {
      logMonitorEl.addEventListener('scroll', () => {
        const atBottom = logMonitorEl.scrollHeight <= (logMonitorEl.scrollTop + logMonitorEl.clientHeight);
        if (atBottom && logMonitorEl.parentElement) logMonitorEl.parentElement.style.cssText = '';
      });
    }
  }
  if (!logMonitorEl) return;
  const atBottom = logMonitorEl.scrollHeight <= (logMonitorEl.scrollTop + logMonitorEl.clientHeight);
  try {
    const res = await authFetch(`${window.api}/log?clear=True`);
    if (res?.ok) {
      logMonitorStatus = true;
      const lines = (await res.json()) as string[];
      if (logMonitorEl && lines?.length > 0 && logMonitorEl.parentElement?.parentElement instanceof HTMLElement) {
        logMonitorEl.parentElement.parentElement.style.display = window.opts.logmonitor_show ? 'block' : 'none';
      }
      for (const line of lines) addLogLine(line);
      if (!logConnected) {
        logConnected = true;
        xhrPost(`${window.api}/log`, { debug: 'connected' });
      }
    } else {
      logConnected = false;
      logErrors++;
      addLogLine(`{ "created": ${Date.now()}, "level":"ERROR", "module":"logMonitor", "facility":"ui", "msg":"Failed to fetch log: ${res?.status} ${res?.statusText}" }`);
    }
    cleanupLog(atBottom);
  } catch {
    logConnected = false;
    logErrors++;
    addLogLine(`{ "created": ${Date.now()}, "level":"ERROR", "module":"logMonitor", "facility":"ui", "msg":"Failed to fetch log: server unreachable" }`);
    cleanupLog(atBottom);
  }
}

export async function initLogMonitor() {
  const el = document.getElementsByTagName('footer')[0];
  if (!el) return;
  const t0 = performance.now();
  el.classList.add('log-monitor');
  const uiDisabled = Array.isArray(window.opts.ui_disabled) ? window.opts.ui_disabled : [];
  if (uiDisabled.includes('logs')) return;
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
  authFetch(`${window.api}/start?agent=${encodeURI(navigator.userAgent)}`);
  logMonitor();
  const t1 = performance.now();
  log('initLogMonitor', { show: window.opts.logmonitor_show, time: Math.round(t1 - t0) });
  timer('initLogMonitor', t1 - t0);
}
