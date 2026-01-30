class ConnectionMonitorState {
  static element;
  static version = '';
  static commit = '';
  static branch = '';
  static online = false;

  static getModel() {
    const cp = opts?.sd_model_checkpoint || '';
    return cp ? this.trimModelName(cp) : 'unknown model';
  }

  static trimModelName(name) {
    // remove trailing [hash], split on / or \, return last segment, trim
    return name.replace(/\s*\[.*\]\s*$/, '').split(/[\\/]/).pop().trim() || 'unknown model';
  }

  static setData({ online, updated, commit, branch }) {
    this.online = online;
    this.version = updated;
    this.commit = commit;
    this.branch = branch;
  }

  static setElement(el) {
    this.element = el;
  }

  static toHTML(modelOverride) {
    return `
      Version: <b>${this.version}</b><br>
      Commit: <b>${this.commit}</b><br>
      Branch: <b>${this.branch}</b><br>
      Status: ${this.online ? '<b style="color:lime">online</b>' : '<b style="color:darkred">offline</b>'}<br>
      Model: <b>${modelOverride ? this.trimModelName(modelOverride) : this.getModel()}</b><br>
      Since: ${new Date().toLocaleString()}<br>
    `;
  }

  static updateState(incomingModel) {
    this.element.dataset.hint = this.toHTML(incomingModel);
    this.element.style.backgroundColor = this.online ? 'var(--sd-main-accent-color)' : 'var(--color-error)';
  }
}

let monitorAutoUpdating = false;

async function updateIndicator(online, data, msg) {
  const el = document.getElementById('logo_nav');
  if (!el || !data) return;
  ConnectionMonitorState.setElement(el);
  if (!monitorAutoUpdating) {
    monitorOption('sd_model_checkpoint', (newVal) => { ConnectionMonitorState.updateState(newVal); }); // Runs before opt actually changes
    monitorAutoUpdating = true;
  }
  ConnectionMonitorState.setData({ online, ...data });
  ConnectionMonitorState.updateState();
  if (online) {
    log('monitorConnection: online', data);
  } else {
    log('monitorConnection: offline', msg);
  }
}

async function monitorConnection() {
  try {
    const res = await authFetch(`${window.api}/version`);
    const data = await res.json();
    const url = res.url.split('/sdapi')[0].replace('http', 'ws'); // update global url as ws need fqdn
    const ws = new WebSocket(`${url}/queue/join`);
    ws.onopen = () => updateIndicator(true, data, '');
    ws.onclose = () => updateIndicator(false, data, '');
    ws.onerror = (e) => updateIndicator(false, data, e.message);
    ws.onmessage = (evt) => log('monitorConnection: message', evt.data);
  } catch { /**/ }
}
