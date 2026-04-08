class ConnectionMonitorState {
  static ws = undefined;
  static delay = 1000;
  static element;
  static version = '';
  static commit = '';
  static branch = '';
  static model = '';
  static startup = '';
  static online = false;

  static getModel() {
    const cp = opts?.sd_model_checkpoint || '';
    return cp ? this.trimModelName(cp) : 'unknown model';
  }

  static trimModelName(name) {
    return name.replace(/\s*\[.*\]\s*$/, '').split(/[\\/]/).pop().trim() || 'unknown model';
  }

  static setData({ online, data }) {
    this.online = online;
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

async function wsMonitorLoop(url) {
  try {
    ConnectionMonitorState.ws = new WebSocket(`${url}/queue/join`);
    ConnectionMonitorState.ws.onopen = () => {};
    ConnectionMonitorState.ws.onmessage = (evt) => updateIndicator(true);
    ConnectionMonitorState.ws.onclose = () => {
      setTimeout(() => wsMonitorLoop(url), ConnectionMonitorState.delay); // happens regularly if there is no traffic
    };
    ConnectionMonitorState.ws.onerror = (e) => {
      updateIndicator(false, {}, e.message); // actual error
      setTimeout(() => wsMonitorLoop(url), ConnectionMonitorState.delay);
    };
  } catch (e) {
    updateIndicator(false, {}, e.message);
    setTimeout(monitorConnection, ConnectionMonitorState.delay); // eslint-disable-line no-use-before-define
  }
}

let monitorActive = false;

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
    updateIndicator(true, data);
    const url = res.url.split('/sdapi')[0].replace('https:', 'wss:').replace('http:', 'ws:'); // update global url as ws need fqdn
    wsMonitorLoop(url);
  } catch {
    updateIndicator(false, data);
    setTimeout(monitorConnection, ConnectionMonitorState.delay);
  }
}
