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
import { initModels, initSettings, loadInitialOpts } from './settings';
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
import { addLegacyNotice } from './legacy';

window.api = '/sdapi/v1';
window.subpath = '';

const startupPromises: Promise<unknown>[] = [];
let optsReady = false;
let initialized = false;

async function waitForOpts() {
  // make sure all of the ui is ready and options are loaded
  const t0 = performance.now();
  let t1 = t0;
  let polls = 0;
  let optsFirstSeen: number | null = null;
  let uiuxSeparatorFirstSeen: number | null = null;
  let pollDelayTotal = 0;
  let pollDelayMax = 0;
  const diagnostics = () => ({
    elapsed: Math.round(t1 - t0),
    polls,
    optsFirstSeen: optsFirstSeen === null ? null : Math.round(optsFirstSeen),
    uiuxSeparatorFirstSeen: uiuxSeparatorFirstSeen === null ? null : Math.round(uiuxSeparatorFirstSeen),
    pollDelayAverage: polls > 0 ? Math.round(pollDelayTotal / polls) : 0,
    pollDelayMax: Math.round(pollDelayMax),
  });
  while (true) {
    if (t1 - t0 > 60000) {
      log('waitForOpts timeout', diagnostics());
      break;
    }
    if (window.opts && Object.keys(window.opts).length > 0) {
      if (optsFirstSeen === null) {
        optsFirstSeen = t1 - t0;
        timer('waitForOpts:options', optsFirstSeen);
      }
      const hasUiuxSeparator = 'uiux_separator_appearance' in window.opts;
      if (hasUiuxSeparator && uiuxSeparatorFirstSeen === null) {
        uiuxSeparatorFirstSeen = t1 - t0;
        timer('waitForOpts:uiux-separator', uiuxSeparatorFirstSeen);
      }
      optsReady = (window.opts.theme_type === 'Modern') ? hasUiuxSeparator : true;
      if (optsReady) {
        log('waitForOpts', diagnostics());
        timer('waitForOpts', t1 - t0);
        break;
      }
    }
    if (t1 - t0 > 15000) {
      log('waitForOpts delayed', Math.round(t1 - t0));
    }
    const sleepStart = performance.now();
    await sleep(50);
    t1 = performance.now();
    const pollDelay = t1 - sleepStart;
    pollDelayTotal += pollDelay;
    pollDelayMax = Math.max(pollDelayMax, pollDelay);
    polls++;
  }
}

async function postStartup() {
  log('postStartup');
  disconnectHintsObserver();
  logTimers();
}

async function updateSubpath() {
  log('mountURL', window.opts.subpath);
  if (window.opts.subpath?.length > 0) {
    window.subpath = window.opts.subpath;
    window.api = `${window.subpath}/sdapi/v1`;
  }
  log('API', { url: window.api });
}

export async function initStartup() {
  if (initialized) return;
  initialized = true;
  const t0 = performance.now();
  log('initGradio', Math.round(t0 - appStartTime));
  timer('initGradio', t0 - appStartTime);
  log('initUi');
  if (window.setupLogger) await window.setupLogger();
  log('loadInitialOpts', { loaded: loadInitialOpts() });

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
  await updateSubpath();

  executeCallbacks(uiReadyCallbacks);

  // optionally wait for modern ui
  if (window.opts.theme_type.toLowerCase().startsWith('modern') && window.waitForUiReady) await window.waitForUiReady();

  // post startup tasks that may take longer but are not critical
  startupPromises.push(Promise.resolve(initLogMonitor()));
  startupPromises.push(Promise.resolve(setRefreshInterval()));
  startupPromises.push(Promise.resolve(setupExtraNetworks()));
  startupPromises.push(Promise.resolve(monitorConnection()));
  startupPromises.push(Promise.resolve(showNetworks()));
  startupPromises.push(Promise.resolve(setHints()));
  startupPromises.push(Promise.resolve(applyStyles()));
  startupPromises.push(Promise.resolve(initIndexDB()));
  startupPromises.push(Promise.resolve(initTableSorter()));
  startupPromises.push(Promise.resolve(initAutocomplete()));

  if (window.opts.theme_type !== 'Modern') addLegacyNotice();

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

window.initStartup = initStartup;
window.addEventListener('pageshow', (evt) => log('pageShow', evt));
window.addEventListener('pagehide', (evt) => log('pageHide', evt));

// onAfterUiUpdate(() => log('evt onAfterUiUpdate'));
// onUiLoaded(() => log('evt onUiLoaded'));
// onOptionsChanged(() => log('evt onOptionsChanged'));
// onUiTabChange(() => log('evt onUiTabChange'));
// onUiUpdate(() => log('evt onUiUpdate'));
