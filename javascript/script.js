async function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms)); // eslint-disable-line no-promise-executor-return
}

function gradioApp() {
  const elems = document.getElementsByTagName('gradio-app');
  const elem = elems.length === 0 ? document : elems[0];
  if (elem !== document) elem.getElementById = (id) => document.getElementById(id);
  return elem.shadowRoot ? elem.shadowRoot : elem;
}

function logFn(func) {
  return async function () { // eslint-disable-line func-names
    const t0 = performance.now();
    const returnValue = func(...arguments);
    const t1 = performance.now();
    log(func.name, `time=${Math.round(t1 - t0)}`);
    return returnValue;
  };
}

function getUICurrentTab() {
  return gradioApp().querySelector('#tabs button.selected');
}

function getUICurrentTabContent() {
  return gradioApp().querySelector('.tabitem[id^=tab_]:not([style*="display: none"])');
}

const get_uiCurrentTabContent = getUICurrentTabContent;
const get_uiCurrentTab = getUICurrentTab;
const uiAfterUpdateCallbacks = [];
const uiUpdateCallbacks = [];
const uiLoadedCallbacks = [];
const uiReadyCallbacks = [];
const uiTabChangeCallbacks = [];
const optionsChangedCallbacks = [];
let uiCurrentTab = null;
let uiAfterUpdateTimeout = null;

function onAfterUiUpdate(callback) {
  uiAfterUpdateCallbacks.push(callback);
}

function onUiUpdate(callback) {
  uiUpdateCallbacks.push(callback);
}

function onUiLoaded(callback) {
  uiLoadedCallbacks.push(callback);
}

function onUiReady(callback) {
  uiReadyCallbacks.push(callback);
}

function onUiTabChange(callback) {
  uiTabChangeCallbacks.push(callback);
}

function onOptionsChanged(callback) {
  optionsChangedCallbacks.push(callback);
}

function executeCallbacks(queue, arg) {
  // if (!uiLoaded) return
  for (const callback of queue) {
    if (!callback) continue;
    try {
      callback(arg);
    } catch (e) {
      error(`executeCallbacks: ${callback} ${e}`);
    }
  }
}

const anyPromptExists = () => gradioApp().querySelectorAll('.main-prompts').length > 0;

function scheduleAfterUiUpdateCallbacks() {
  clearTimeout(uiAfterUpdateTimeout);
  uiAfterUpdateTimeout = setTimeout(() => executeCallbacks(uiAfterUpdateCallbacks, 500));
}

let executedOnLoaded = false;
const ignoreElements = ['logMonitorData', 'logWarnings', 'logErrors', 'tooltip-container', 'logger'];
const ignoreClasses = ['wrap'];

let mutationTimer = null;
let validMutations = [];
async function mutationCallback(mutations) {
  let newMutations = mutations;
  if (newMutations.length > 0) newMutations = newMutations.filter((m) => m.target.nodeName !== 'LABEL');
  if (newMutations.length > 0) newMutations = newMutations.filter((m) => ignoreElements.indexOf(m.target.id) === -1);
  if (newMutations.length > 0) newMutations = newMutations.filter((m) => m.target.id !== 'logWarnings' && m.target.id !== 'logErrors');
  if (newMutations.length > 0) newMutations = newMutations.filter((m) => !m.target.classList?.contains('wrap'));
  if (newMutations.length > 0) validMutations = validMutations.concat(newMutations);
  if (validMutations.length < 1) return;

  if (mutationTimer) clearTimeout(mutationTimer);
  mutationTimer = setTimeout(async () => {
    if (!executedOnLoaded && anyPromptExists()) { // execute once
      executedOnLoaded = true;
      executeCallbacks(uiLoadedCallbacks);
    }
    if (executedOnLoaded) { // execute on each mutation
      executeCallbacks(uiUpdateCallbacks, mutations);
      scheduleAfterUiUpdateCallbacks();
    }
    const newTab = getUICurrentTab();
    if (newTab && (newTab !== uiCurrentTab)) {
      uiCurrentTab = newTab;
      executeCallbacks(uiTabChangeCallbacks);
    }
    validMutations = [];
    mutationTimer = null;
  }, 50);
}

document.addEventListener('DOMContentLoaded', () => {
  const mutationObserver = new MutationObserver(mutationCallback);
  mutationObserver.observe(gradioApp(), { childList: true, subtree: true });
});

/**
 * Add a listener to the document for keydown events
 */
document.addEventListener('keydown', (e) => {
  let elem;
  if (e.key === 'Escape') elem = getUICurrentTabContent().querySelector('button[id$=_interrupt]');
  if (e.key === 'Enter' && e.ctrlKey) elem = getUICurrentTabContent().querySelector('button[id$=_generate]');
  if (e.key === 'i' && e.ctrlKey) elem = getUICurrentTabContent().querySelector('button[id$=_reprocess]');
  if (e.key === ' ' && e.ctrlKey) elem = getUICurrentTabContent().querySelector('button[id$=_extra_networks_btn]');
  if (e.key === 'n' && e.ctrlKey) elem = getUICurrentTabContent().querySelector('button[id$=_extra_networks_btn]');
  if (e.key === 's' && e.ctrlKey) elem = getUICurrentTabContent().querySelector('button[id^=save_]');
  if (e.key === 'Insert' && e.ctrlKey) elem = getUICurrentTabContent().querySelector('button[id^=save_]');
  if (e.key === 'd' && e.ctrlKey) elem = getUICurrentTabContent().querySelector('button[id^=delete_]');
  // if (e.key === 'm' && e.ctrlKey) elem = gradioApp().getElementById('setting_sd_model_checkpoint');
  if (elem) {
    e.preventDefault();
    log('hotkey', { key: e.key, meta: e.metaKey, ctrl: e.ctrlKey, alt: e.altKey }, elem?.id, elem.nodeName);
    if (elem.nodeName === 'BUTTON') elem.click();
    else elem.focus();
  }
});

/**
 * checks that a UI element is not in another hidden element or tab content
 */
function uiElementIsVisible(el) {
  if (el === document) return true;
  const computedStyle = getComputedStyle(el);
  const isVisible = computedStyle.display !== 'none';
  if (!isVisible) return false;
  return uiElementIsVisible(el.parentNode);
}

function uiElementInSight(el) {
  const clRect = el.getBoundingClientRect();
  const windowHeight = window.innerHeight;
  const isOnScreen = clRect.bottom > 0 && clRect.top < windowHeight;
  return isOnScreen;
}
