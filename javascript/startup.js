/* eslint-disable no-undef */
window.api = '/sdapi/v1';
window.subpath = '';

const startupPromises = [];

async function waitForOpts() {
  // make sure all of the ui is ready and options are loaded
  const t0 = performance.now();
  let t1 = performance.now();
  while (true) {
    if (t1 - t0 > 120000) {
      log('waitForOpts timeout');
      break;
    }
    if (window.opts && Object.keys(window.opts).length > 0) {
      ok = window.opts.theme_type === 'Modern' ? 'uiux_separator_appearance' in window.opts : true;
      if (ok) {
        log('waitForOpts', Math.round(t1 - t0));
        timer('waitForOpts', t1 - t0);
        break;
      }
    }
    await sleep(100);
    t1 = performance.now();
  }
}

async function postStartup() {
  log('postStartup');
  if (window.gradioObserver) window.gradioObserver.disconnect();
  if (window.hintsObserver) window.hintsObserver.disconnect();
  logTimers();
}

async function initStartup() {
  const t0 = performance.now();
  log('initGradio', Math.round(t0 - appStartTime));
  timer('initGradio', t0 - appStartTime);
  log('initUi');
  if (window.setupLogger) await setupLogger();

  // all items here are non-blocking async calls

  startupPromises.push(initModels());
  startupPromises.push(getUIDefaults());
  startupPromises.push(initPromptChecker());
  startupPromises.push(initContextMenu());
  startupPromises.push(initDragDrop());
  startupPromises.push(initAccordions());
  startupPromises.push(initSettings());
  startupPromises.push(initImageViewer());
  startupPromises.push(initiGenerationParams());
  startupPromises.push(initChangelog());
  startupPromises.push(setupControlUI());

  // reconnect server session
  await reconnectUI();
  await waitForOpts();

  log('mountURL', window.opts.subpath);
  if (window.opts.subpath?.length > 0) {
    window.subpath = window.opts.subpath;
    window.api = `${window.subpath}/sdapi/v1`;
  }

  executeCallbacks(uiReadyCallbacks);
  startupPromises.push(initGallery());
  startupPromises.push(setRefreshInterval());
  startupPromises.push(setupExtraNetworks());

  // optinally wait for modern ui
  if (window.waitForUiReady) await waitForUiReady();

  // post startup tasks that may take longer but are not critical
  startupPromises.push(initAutocomplete());
  startupPromises.push(monitorConnection());
  startupPromises.push(showNetworks());
  startupPromises.push(setHints());
  startupPromises.push(applyStyles());
  startupPromises.push(initIndexDB());
  startupPromises.push(initLogMonitor());

  t1 = performance.now();
  log('initStartup', Math.round(1000 * (t1 - t0) / 1000000));

  removeSplash();

  await Promise.all(startupPromises);
  t2 = performance.now();
  log('initComplete', Math.round(1000 * (t2 - t0) / 1000000));
  postStartup();
}

onUiLoaded(initStartup);
onUiReady(() => log('uiReady'));

// onAfterUiUpdate(() => log('evt onAfterUiUpdate'));
// onUiLoaded(() => log('evt onUiLoaded'));
// onOptionsChanged(() => log('evt onOptionsChanged'));
// onUiTabChange(() => log('evt onUiTabChange'));
// onUiUpdate(() => log('evt onUiUpdate'));
