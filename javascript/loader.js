const appStartTime = performance.now();
let monitorLogActive = false;

async function preloadImages() {
  const dark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  const imagePromises = [];
  const num = Math.floor(9.99 * Math.random());
  const imageUrls = [
    `file=html/logo-bg-${dark ? 'dark' : 'light'}.jpg`,
    `file=html/logo-bg-${num}.jpg`,
  ];
  for (const url of imageUrls) {
    const img = new Image();
    const promise = new Promise((resolve, reject) => {
      img.onload = resolve;
      img.onerror = reject;
    });
    img.src = url;
    imagePromises.push(promise);
  }
  try {
    await Promise.all(imagePromises);
    return true;
  } catch (err) {
    error(`preloadImages: ${err}`);
    return false;
  }
}

function joinArgs(messages) {
  let output = '';
  for (let i = 0; i < messages.length; i++) {
    let arg = messages[i];
    if (arg === undefined) arg = 'undefined';
    if (arg === null) arg = 'null';
    output += ' ';
    if (typeof arg === 'object') output += JSON.stringify(arg).replace(/["]+/g, '');
    else output += arg;
  }
  return output;
}

async function monitorLog() {
  if (window.logBufferDirty) {
    window.logBufferDirty = false;
    const maxLines = 100; // print last n logs from ring buffer to splash-log
    const lines = [];
    // print last n logs from ring buffer in time order
    for (let i = Math.max(0, window.logRingBuffer.length - maxLines); i < window.logRingBuffer.length; i++) {
      const logEntry = window.logRingBuffer[i];
      let color = 'white';
      if (logEntry.type === 'error') color = 'palevioletred';
      else if (logEntry.type === 'debug') color = 'gray';
      const html = `<div class="splash-log-row" style="color: ${color}">${logEntry.ts} &nbsp; ${joinArgs(logEntry.msg)}</div>`;
      lines.push(html);
    }
    const splashLogEl = document.getElementById('splashLog');
    if (splashLogEl) splashLogEl.innerHTML = lines.join('');
  }
  if (monitorLogActive) setTimeout(monitorLog, 250);
}

async function removeSplash() {
  const splash = document.getElementById('splash');
  if (splash) splash.remove();
  log('removeSplash');
  const t = Math.round(performance.now() - appStartTime);
  log('startupTime', t);
  timer('splashVisible', t);
  xhrPost(`${window.api}/log`, { message: `ready time=${t}` });
  monitorLogActive = false;
}

async function createSplash() {
  const dark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  log('createSplash', { theme: dark ? 'dark' : 'light' });
  const num = Math.floor(9.99 * Math.random());
  const splash = `
    <div id="splash" class="splash" style="background: ${dark ? 'black' : 'white'}">
      <div class="loading"><div class="loader"></div></div>
      <div id="motd" class="motd""></div>
      <div id="splashLog" class="splash-log" style="position: fixed; bottom: 0; text-align: left; padding: 8vh 8px 8px 8px; font-size: 12px; width: 100%; background: linear-gradient(0deg, darkslategray, transparent); opacity: 50%;"></div>
    </div>`;
  document.body.insertAdjacentHTML('beforeend', splash);
  const ok = await preloadImages();
  if (!ok) {
    removeSplash();
    return;
  }
  const imgEl = `<div id="spash-img" class="splash-img" alt="logo" style="background-image: url(file=html/logo-bg-${dark ? 'dark' : 'light'}.jpg), url(file=html/logo-bg-${num}.jpg); background-blend-mode: ${dark ? 'multiply' : 'lighten'}"></div>`;
  const splashEl = document.getElementById('splash');
  if (splashEl) splashEl.insertAdjacentHTML('afterbegin', imgEl);

  monitorLogActive = true;
  monitorLog();

  await authFetch(`${window.api}/motd`)
    .then((res) => res.text())
    .then((text) => {
      const clean = text.replace(/["]+/g, '');
      log('getMOTD', clean);
      const motdEl = document.getElementById('motd');
      if (motdEl) motdEl.innerHTML = clean;
    })
    .catch((err) => error(`getMOTD: ${err}`));

  log('loadGradioUi');
}

window.onload = createSplash;
