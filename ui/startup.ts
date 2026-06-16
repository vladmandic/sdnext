import { initChangelog } from './changelog';
import { setupControlUI } from './control';
import { initiGenerationParams } from './generationParams';
import { initDragDrop } from './imageParams';
import { initAccordions } from './inputAccordion';
import { initIndexDB } from './indexdb';
import { initLogMonitor } from './logMonitor';
import { monitorConnection } from './monitor';
import { setRefreshInterval } from './progressBar';
import { initPromptChecker } from './promptChecker';
import { initModels, initSettings } from './settings';
import { initGallery } from './gallery';
import { initImageViewer } from './imageViewer';
import { reconnectUI } from './ui';
import { setupExtraNetworks, showNetworks, applyStyles } from './extraNetworks';
import { initAutocomplete } from './autocomplete';
import { setHints, disconnectHintsObserver } from './setHints';
import { initContextMenu } from './contextMenus';
import { executeCallbacks, onUiLoaded, onUiReady, sleep, uiReadyCallbacks, initTableSorter } from './script';
import { timer, logTimers } from './timers';
import { getUIDefaults } from './uiConfig';
import { log } from './logger';
import { appStartTime, removeSplash } from './loader';

window.api = '/sdapi/v1';
window.subpath = '';

const startupPromises: Promise<unknown>[] = [];
let ok = false;

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
  disconnectHintsObserver();
  logTimers();
}

async function initStartup() {
  const t0 = performance.now();
  log('initGradio', Math.round(t0 - appStartTime));
  timer('initGradio', t0 - appStartTime);
  log('initUi');
  if (window.setupLogger) await window.setupLogger();

  // all items here are non-blocking async calls

  startupPromises.push(initModels());
  startupPromises.push(getUIDefaults());
  startupPromises.push(initPromptChecker());
  startupPromises.push(initContextMenu());
  startupPromises.push(initDragDrop());
  startupPromises.push(Promise.resolve(initAccordions()));
  startupPromises.push(Promise.resolve(initSettings()));
  startupPromises.push(Promise.resolve(initImageViewer()));
  startupPromises.push(Promise.resolve(initGallery()));
  startupPromises.push(Promise.resolve(initiGenerationParams()));
  startupPromises.push(Promise.resolve(initChangelog()));
  startupPromises.push(Promise.resolve(setupControlUI()));

  // reconnect server session
  await reconnectUI();
  await waitForOpts();

  log('mountURL', window.opts.subpath);
  if (window.opts.subpath?.length > 0) {
    window.subpath = window.opts.subpath;
    window.api = `${window.subpath}/sdapi/v1`;
  }

  startupPromises.push(initLogMonitor());

  executeCallbacks(uiReadyCallbacks);

  // optionally wait for modern ui
  if (window.waitForUiReady) await window.waitForUiReady();

  // post startup tasks that may take longer but are not critical
  startupPromises.push(Promise.resolve(initGallery()));
  startupPromises.push(Promise.resolve(setRefreshInterval()));
  startupPromises.push(Promise.resolve(setupExtraNetworks()));
  startupPromises.push(Promise.resolve(initAutocomplete()));
  startupPromises.push(Promise.resolve(monitorConnection()));
  startupPromises.push(Promise.resolve(showNetworks()));
  startupPromises.push(Promise.resolve(setHints()));
  startupPromises.push(Promise.resolve(applyStyles()));
  startupPromises.push(Promise.resolve(initIndexDB()));
  startupPromises.push(Promise.resolve(initTableSorter()));

  const t1 = performance.now();
  log('initStartup', Math.round(1000 * (t1 - t0) / 1000000));

  removeSplash();

  await Promise.all(startupPromises);
  const t2 = performance.now();
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
