import { debug, log } from './logger';
import { authFetch } from './authWrap';
import { monitorOption } from './settings';

interface VersionInfo {
  updated?: string;
  commit?: string;
  branch?: string;
  model?: string;
}

let monitorActive = false;
let wsTimer: ReturnType<typeof setTimeout> | undefined;

export class ConnectionMonitorState {
  static ws: WebSocket | undefined;
  static url = '';
  static delay = 2000;
  static element: HTMLElement | undefined;
  static version = '';
  static commit = '';
  static branch = '';
  static model = '';
  static startup: Date = new Date();
  static online = false;
  static ts: Date = new Date();

  static getModel(): string {
    const cp = window.opts?.sd_model_checkpoint || '';
    return cp ? this.trimModelName(cp) : 'unknown model';
  }

  static trimModelName(name: string): string {
    return name.replace(/\s*\[.*\]\s*$/, '').split(/[\\/]/).pop().trim() || 'unknown model';
  }

  static setData({ online, data }: { online: boolean; data: VersionInfo }) {
    if (online !== this.online) {
      this.online = online;
      this.ts = new Date();
      debug('monitorState', { online: ConnectionMonitorState.online, ts: ConnectionMonitorState.ts?.toLocaleTimeString() });
    }
    if (data?.updated) this.version = data.updated;
    if (data?.commit) this.commit = data.commit;
    if (data?.branch) this.branch = data.branch;
    if (data?.model) this.model = this.trimModelName(data.model);
  }

  static toHTML(): string {
    if (!this.model) this.model = this.getModel();
    return `
      Version: <b>${this.version}</b><br>
      Commit: <b>${this.commit}</b><br>
      Branch: <b>${this.branch}</b><br>
      Status: ${this.online ? '<b style="color:lime">online</b>' : '<b style="color:darkred">offline</b>'}<br>
      Model: <b>${this.model}</b><br>
      Since: ${this.startup.toLocaleString()}<br>
    `;
  }

  static updateState() {
    if (!this.element) {
      const el = document.getElementById('logo_nav');
      if (el) this.element = el;
      else return;
    }
    this.element.dataset.hint = this.toHTML();
    this.element.style.backgroundColor = this.online ? 'var(--sd-main-accent-color)' : 'var(--color-error)';
  }
}

async function updateIndicator(online: boolean, data: VersionInfo = {}, msg?: string): Promise<void> {
  ConnectionMonitorState.setData({ online, data });
  ConnectionMonitorState.updateState();
  if (msg) log('monitorConnection:', { online, data, msg });
}

function scheduleNextLoop() {
  if (wsTimer) {
    clearTimeout(wsTimer);
    wsTimer = undefined;
  }
  const offlineDurationMs = Date.now() - ConnectionMonitorState.ts.getTime();
  if (!ConnectionMonitorState.online && offlineDurationMs > (60 * 60 * 1000)) ConnectionMonitorState.delay = 10000;
  else if (!ConnectionMonitorState.online && offlineDurationMs > (5 * 60 * 1000)) ConnectionMonitorState.delay = 5000;
  else ConnectionMonitorState.delay = 1000;
  wsTimer = setTimeout(wsMonitorLoop, ConnectionMonitorState.delay); // eslint-disable-line @typescript-eslint/no-use-before-define
}

async function wsMonitorLoop() {
  // Tear down any existing socket before creating a new one
  if (ConnectionMonitorState.ws) {
    ConnectionMonitorState.ws.onopen = null;
    ConnectionMonitorState.ws.onmessage = null;
    ConnectionMonitorState.ws.onclose = null;
    ConnectionMonitorState.ws.onerror = null;
    try {
      ConnectionMonitorState.ws.close();
    } catch {
      // Ignore cleanup errors on stale sockets
    }
    ConnectionMonitorState.ws = undefined;
  }

  try {
    ConnectionMonitorState.ws = new WebSocket(`${ConnectionMonitorState.url}/internal/monitor`);
    ConnectionMonitorState.ws.onopen = () => updateIndicator(true);
    ConnectionMonitorState.ws.onmessage = (msg: MessageEvent) => updateIndicator(true, msg.data ? JSON.parse(msg.data) : {});
    ConnectionMonitorState.ws.onclose = () => {
      updateIndicator(false);
      scheduleNextLoop();
    };
    ConnectionMonitorState.ws.onerror = (e: Event) => updateIndicator(false, {}, String((e as ErrorEvent).message || 'unknown error'));
  } catch (e) {
    updateIndicator(false, {}, String((e as Error).message || e));
    scheduleNextLoop();
  }
}

export async function monitorConnection() {
  if (!monitorActive) { // start monitor loop only once on startup
    monitorActive = true;
    monitorOption('sd_model_checkpoint', (newVal) => { // runs before opt actually changes
      ConnectionMonitorState.model = newVal;
      ConnectionMonitorState.updateState();
    });
  }
  ConnectionMonitorState.startup = new Date();

  let data: VersionInfo = {};
  try {
    const res = await authFetch(`${window.api}/version`);
    if (!res) throw new Error('No response');
    data = await res.json();
    log('monitorConnection:', { data });
    ConnectionMonitorState.startup = new Date();
    ConnectionMonitorState.url = res.url.split('/sdapi')[0].replace('https:', 'wss:').replace('http:', 'ws:'); // update global url as ws need fqdn
    updateIndicator(true, data);
    wsMonitorLoop();
  } catch {
    updateIndicator(false, data);
    scheduleNextLoop();
  }
}
