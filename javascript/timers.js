const allTimers = [];

async function timer(name, elapsed) {
  allTimers.push([name, Math.round(elapsed)]);
}

async function logTimers() {
  allTimers.sort((a, b) => b[1] - a[1]);
  const filteredTimers = allTimers.filter((t) => t[1] > 50);
  log('timers', filteredTimers);
  // xhrPost(`${window.api}/log`, { debug: JSON.stringify(filteredTimers) });
}

window.timer = timer;
