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

export class ConnectionMonitorState {
  static ws: WebSocket | undefined;
  static url = '';
  static delay = 1000;
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
      debug('monitorState', { online: ConnectionMonitorState.online, ts: ConnectionMonitorState.ts });
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

async function wsMonitorLoop() {
  const delayed = Date.now() - ConnectionMonitorState.ts.getTime();
  if ((delayed > 60 * 60) && (ConnectionMonitorState.delay < 10) && !ConnectionMonitorState.online) ConnectionMonitorState.delay = 10000;
  else if ((delayed > 5 * 60) && (ConnectionMonitorState.delay < 5) && !ConnectionMonitorState.online) ConnectionMonitorState.delay = 5000;
  else ConnectionMonitorState.delay = 2000;
  try {
    ConnectionMonitorState.ws = new WebSocket(`${ConnectionMonitorState.url}/queue/join`);
    ConnectionMonitorState.ws.onopen = () => {};
    ConnectionMonitorState.ws.onmessage = () => updateIndicator(true);
    ConnectionMonitorState.ws.onclose = () => setTimeout(wsMonitorLoop, ConnectionMonitorState.delay); // main re-check loop
    ConnectionMonitorState.ws.onerror = (e: Event) => updateIndicator(false, {}, String((e as ErrorEvent).message || 'unknown error')); // actual error
  } catch (e) {
    updateIndicator(false, {}, String((e as Error).message || e));
    // eslint-disable-next-line @typescript-eslint/no-use-before-define
    setTimeout(monitorConnection, ConnectionMonitorState.delay);
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
    setTimeout(monitorConnection, ConnectionMonitorState.delay);
  }
}
