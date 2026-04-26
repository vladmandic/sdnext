let monitorActive = false;

class ConnectionMonitorState {
  static ws = undefined;
  static url = '';
  static delay = 1000;
  static element;
  static version = '';
  static commit = '';
  static branch = '';
  static model = '';
  static startup = '';
  static online = false;
  static ts = Date();

  static getModel() {
    const cp = opts?.sd_model_checkpoint || '';
    return cp ? this.trimModelName(cp) : 'unknown model';
  }

  static trimModelName(name) {
    return name.replace(/\s*\[.*\]\s*$/, '').split(/[\\/]/).pop().trim() || 'unknown model';
  }

  static setData({ online, data }) {
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

  static toHTML() {
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

async function updateIndicator(online, data = {}, msg = undefined) {
  ConnectionMonitorState.setData({ online, data });
  ConnectionMonitorState.updateState();
  if (msg) log('monitorConnection:', { online, data, msg });
}

async function wsMonitorLoop() {
  const delayed = new Date() - ConnectionMonitorState.ts;
  if ((delayed > 60 * 60) && (ConnectionMonitorState.delay < 10) && !ConnectionMonitorState.online) ConnectionMonitorState.delay = 10000;
  else if ((delayed > 5 * 60) && (ConnectionMonitorState.delay < 5) && !ConnectionMonitorState.online) ConnectionMonitorState.delay = 5000;
  else ConnectionMonitorState.delay = 2000;
  try {
    ConnectionMonitorState.ws = new WebSocket(`${ConnectionMonitorState.url}/queue/join`);
    ConnectionMonitorState.ws.onopen = () => {};
    ConnectionMonitorState.ws.onmessage = (evt) => updateIndicator(true);
    ConnectionMonitorState.ws.onclose = () => setTimeout(wsMonitorLoop, ConnectionMonitorState.delay); // main re-check loop
    ConnectionMonitorState.ws.onerror = (e) => updateIndicator(false, {}, e.message); // actual error
  } catch (e) {
    updateIndicator(false, {}, e.message);
    setTimeout(monitorConnection, ConnectionMonitorState.delay); // eslint-disable-line no-use-before-define
  }
}

async function monitorConnection() {
  if (!monitorActive) { // start monitor loop only once on startup
    monitorActive = true;
    monitorOption('sd_model_checkpoint', (newVal) => { // runs before opt actually changes
      ConnectionMonitorState.model = newVal;
      ConnectionMonitorState.updateState();
    });
  }
  ConnectionMonitorState.startup = new Date();

  let data = {};
  try {
    const res = await authFetch(`${window.api}/version`);
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
