async function updateIndicator(online, data, msg) {
  const el = document.getElementById('logo_nav');
  if (!el || !data) return;
  const status = online ? '<b style="color:lime">online</b>' : '<b style="color:darkred">offline</b>';
  const date = new Date();
  const template = `
    Version: <b>${data.updated}</b><br>
    Commit: <b>${data.hash}</b><br>
    Branch: <b>${data.branch}</b><br>
    Status: ${status}<br>
    Since: ${date.toLocaleString()}<br>
  `;
  if (online) {
    el.dataset.hint = template;
    el.style.backgroundColor = 'var(--sd-main-accent-color)';
    log('monitorConnection: online', data);
  } else {
    el.dataset.hint = template;
    el.style.backgroundColor = 'var(--color-error)';
    log('monitorConnection: offline', msg);
  }
}

async function monitorConnection() {
  try {
    const res = await fetch(`${window.api}/version`);
    const data = await res.json();
    const url = res.url.split('/sdapi')[0].replace('http', 'ws'); // update global url as ws need fqdn
    const ws = new WebSocket(`${url}/queue/join`);
    ws.onopen = () => updateIndicator(true, data, '');
    ws.onclose = () => updateIndicator(false, data, '');
    ws.onerror = (e) => updateIndicator(false, data, e.message);
    ws.onmessage = (evt) => log('monitorConnection: message', evt.data);
  } catch { /**/ }
}
