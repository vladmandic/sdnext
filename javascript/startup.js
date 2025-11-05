/* eslint-disable no-undef */
window.api = '/sdapi/v1';
window.subpath = '';

async function waitForOpts() {
  // make sure all of the ui is ready and options are loaded
  const t0 = performance.now();
  let t1 = performance.now();
  while (true) { // eslint-disable-line no-constant-condition
    if (t1 - t0 > 120000) {
      log('waitForOpts timeout');
      break;
    }
    if (window.opts && Object.keys(window.opts).length > 0) {
      ok = window.opts.theme_type === 'Modern' ? 'uiux_separator_appearance' in window.opts : true;
      if (ok) {
        log('waitForOpts', `time=${Math.round(t1 - t0)}`);
        break;
      }
    }
    await sleep(50);
    t1 = performance.now();
  }
}

async function initStartup() {
  const t0 = performance.now();
  log('gradio', `time=${Math.round(t0 - appStartTime)}`);
  log('initStartup');
  if (window.setupLogger) await setupLogger();

  // all items here are non-blocking async calls
  await initModels();
  await getUIDefaults();
  await initPromptChecker();
  await initContextMenu();
  await initDragDrop();
  await initAccordions();
  await initSettings();
  await initImageViewer();
  await initGallery();
  await initiGenerationParams();
  await initChangelog();
  await setupControlUI();

  // reconnect server session
  await reconnectUI();
  await waitForOpts();

  log('mountURL', window.opts.subpath);
  if (window.opts.subpath?.length > 0) {
    window.subpath = window.opts.subpath;
    window.api = `${window.subpath}/sdapi/v1`;
  }
  setRefreshInterval();
  executeCallbacks(uiReadyCallbacks);
  initLogMonitor();
  setupExtraNetworks();

  // optinally wait for modern ui
  if (window.waitForUiReady) await waitForUiReady();
  removeSplash();

  // post startup tasks that may take longer but are not critical
  showNetworks();
  setHints();
  applyStyles();
  initIndexDB();
  t1 = performance.now();
  log('initStartup', Math.round(1000 * (t1 - t0) / 1000000));
}

onUiLoaded(initStartup);
onUiReady(() => log('uiReady'));

// onAfterUiUpdate(() => log('evt onAfterUiUpdate'));
// onUiLoaded(() => log('evt onUiLoaded'));
// onOptionsChanged(() => log('evt onOptionsChanged'));
// onUiTabChange(() => log('evt onUiTabChange'));
// onUiUpdate(() => log('evt onUiUpdate'));
